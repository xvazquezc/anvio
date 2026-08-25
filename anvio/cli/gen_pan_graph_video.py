#!/usr/bin/env python

"""Render a pangenome-graph 'growth' video directly from a pan-graph-db.

This program drives `anvio.interactive.PangraphInteractive`, the same backend class that
serves the interactive interface, in a loop: it asks for the pangenome graph of a growing
genome subset (genome 1; genomes 1 and 2; genomes 1 through 3; and so on). Each of those
calls hands back that frame's nodes, edges, and layout positions as plain Python data
structures, with no browser and no SVG parsing involved. Each frame then becomes an
abstract 'scene' (shapes and text, in canvas pixel coordinates), which is rendered twice:
once with PIL for video frames and PNG stills, once as plain SVG for vector stills. Both
renderers walk the very same scene, which is what guarantees that an exported still matches
its video frame exactly.
"""

import os
import sys
import math
import shutil
import hashlib
import tempfile
import argparse
import threading
import importlib.util
import concurrent.futures

import anvio
import anvio.utils as utils
import anvio.terminal as terminal
import anvio.filesnpaths as filesnpaths

from anvio.argparse import ArgumentParser
from anvio.errors import ConfigError, FilesNPathsError

__copyright__ = "Copyleft 2015-2026, The Anvi'o Project (http://anvio.org/)"
__license__ = "GPL 3.0"
__version__ = anvio.__version__
__authors__ = ['meren']
__requires__ = ['pan-graph-db']
__provides__ = []
__description__ = ("Render an MP4 'growth' animation of a pangenome graph directly from a "
                   "pan-graph-db by recomputing the graph's layout for growing genome "
                   "subsets through the same PangraphInteractive backend `anvi-display-pan-graph` "
                   "uses.")


class NewNodeEffect:
    """Base for what --main-graph-new-node-effect and --inset-graph-new-node-effect can ask for
    by name: a brief flourish marking the nodes a genome brought with it.

    A subclass names itself, says what share of --seconds-per-genome it plays for, and
    implements `shapes()`. Everything it needs to be tuned with belongs to IT, as class
    attributes, so no two effects can reach into each other's settings and adding one touches
    nothing but its own class and the registry at the bottom of this block.

    `duration_fraction` under 1.0 leaves the rest of a genome's turn on screen as a still hold.
    At 1.0 and above the effect fills the whole turn and lengthens it, since a genome is always
    held for at least one frame afterwards."""

    name = None
    duration_fraction = 0.75

    def shapes(self, x, y, radius, fill, outline, outline_width, progress):
        """Shapes for one new node, to lay over the frame it appeared in. `x`, `y`, `radius` and
        the colors are how that node itself was drawn, and `progress` runs from 0, the frame the
        node appears, to 1, the end of the effect's span."""

        raise NotImplementedError


    @staticmethod
    def ease_out(t):
        """`t` (0 to 1) redistributed so it moves fast to begin with and eases to a stop."""

        return 1.0 - (1.0 - t) ** 2


class DropletEffect(NewNodeEffect):
    """A copy of the node breaking outwards from where it appeared and fading as it spreads,
    the way a drop does where it lands.

    Neither the spread nor the fade runs at a constant rate. The radius eases to a stop rather
    than marching outwards, and the fade is front-loaded, because an opacity falling away
    linearly reads as mechanical and leaves a tail lingering as a smudge long after the eye has
    lost interest in it."""

    name = 'droplets'
    duration_fraction = 1.2

    max_scale = 20.0        # how far the droplet spreads, as a multiple of the node's radius
    fade_exponent = 2.5     # raise to fade harder at the start and trail off sooner

    def shapes(self, x, y, radius, fill, outline, outline_width, progress):
        scale = 1.0 + (self.max_scale - 1.0) * self.ease_out(progress)
        opacity = (1.0 - progress) ** self.fade_exponent

        return [('fading_circle', x, y, radius * scale, fill, outline, outline_width, opacity)]


# what the two --*-new-node-effect flags accept by name. Effects hold no state, so one instance
# of each is all anyone needs
NEW_NODE_EFFECTS = {effect.name: effect() for effect in [DropletEffect]}


run = terminal.Run()
progress = terminal.Progress()
pp = terminal.pretty_print
P = terminal.pluralize

# what an SVG names as its font-family when the chosen typeface is not on this machine, and
# there is therefore no one family that can honestly be named
FALLBACK_FONT_FAMILY = 'Helvetica, Arial, sans-serif'

# face index -> (font file, {role: face index}, SVG font-family). Every text position
# and text size in this program is authored against a 2160-tall frame, and scaled to the
# actual canvas height by `px()`.
FONT_SETS = {
    'helvetica': ('/System/Library/Fonts/HelveticaNeue.ttc',
                  {'bold': 1, 'demi': 10, 'medium': 10, 'italic': 2, 'regular': 0},
                  'Helvetica Neue'),
    'avenir': ('/System/Library/Fonts/Avenir Next.ttc',
               {'bold': 0, 'demi': 2, 'medium': 5, 'italic': 4, 'regular': 7},
               'Avenir Next'),
}
BOLD_ROLES = {'bold'}
ITALIC_ROLES = {'italic'}

INK, MUTED, FAINT = (17, 17, 17), (140, 140, 140), (228, 228, 228)
BLUE = (31, 111, 235)
TINT_ALPHA = 0.10  # blue wash behind the magnified inset panel

# stands in for a cosmetic value that was left to the pan-graph-db's own state
AUTO_FROM_DB = 'auto (from pan-graph-db)'

# corner radius of the inset panel and of the marker that shows where it came from, in
# 2160-tall-frame units. Enough to read as a rounded box, not enough to read as a pill
INSET_CORNER_RADIUS = 24

# the vertical band, in 2160-tall-frame units, that the inset panel is allowed to occupy
# inside the left text column
PANEL_TOP, PANEL_BOTTOM = 980, 2080

# how many pixels of arc each sampled point along a constant-radius segment covers. Small
# enough that a long sweep around a ring reads as a smooth curve rather than a polygon
ARC_SAMPLE_PX = 4

# the one gap, in 2160-tall-frame units, repeated between every piece of the counter block:
# counter line to progress bar, bar to inset description, description to inset panel
COUNTER_GAP = 44

# --main-graph-nodes-fly-in: a new node arriving from somewhere off-canvas rather than simply
# being there in the next frame. See `fly_in_shapes`.
MAIN_GRAPH_FLY_IN_FRACTION = 1.0        # a node's flight, as a share of --seconds-per-genome
MAIN_GRAPH_FLY_IN_START_DISTANCE = 0.9  # where it sets off, as a multiple of the canvas diagonal
MAIN_GRAPH_FLY_IN_START_SIZE = 3.0      # how big it sets off, as a multiple of its own final size
MAIN_GRAPH_FLY_IN_FADE = False          # fade a node up from nothing over the course of its flight.
                                        # OFF, because a node only a pixel or two across is already
                                        # hard enough to follow without also being half transparent
                                        # for most of the way in. Flip to True to get the fade back

# how far the text trailing the big genome count is lifted off that count's baseline, as a
# fraction of the count's own size. Purely optical: a caption resting exactly on the baseline
# of a numeral that large reads as though it were sliding off it
COUNTER_TEXT_LIFT = 0.065


class PanGraphVideoGenerator:
    """Turns a pan-graph-db into a growth-animation MP4.

    >>> import argparse
    >>> args = argparse.Namespace(pan_graph_db="PATH/TO/PAN-GRAPH.db", output_file="growth.mp4")
    >>> v = PanGraphVideoGenerator(args)
    >>> v.process()
    """

    def __init__(self, args, run=run, progress=progress):
        self.args = args
        self.run = run
        self.progress = progress

        A = lambda x: args.__dict__[x] if x in args.__dict__ else None

        self.pan_graph_db_path = A('pan_graph_db')
        self.output_file = A('output_file')
        self.component = A('component') or 'CP_0001'
        self.genomes_of_interest_arg = A('genomes_of_interest')
        self.exclude_genomes_arg = A('exclude_genomes')
        self.max_num_genomes_to_render = A('max_num_genomes_to_render')

        self.inset_flanks_arg = A('inset_flanks')
        self.inset_nodes_cover_window_dynamically = A('inset_graph_nodes_cover_window_dynamically') or False
        self.inset_aspect = A('inset_aspect') or 1.436

        self.resolution_arg = A('resolution') or '3840x2160'
        self.rotation_offset_deg = A('rotation_offset_deg') or 0.0
        self.graph_height_fraction = A('graph_height_fraction') if A('graph_height_fraction') is not None else 0.2
        self.inner_radius_fraction = A('inner_radius_fraction') if A('inner_radius_fraction') is not None else 0.15
        self.outer_radius_fraction = A('outer_radius_fraction') if A('outer_radius_fraction') is not None else 0.97
        self.track_height_px = A('track_height_px')
        self.backbone_gap_px = A('backbone_gap_px')
        self.supersample = A('supersample') or 2
        self.num_threads = A('num_threads') or 1

        # COSMETICS. Each of these resolves to a default that is computed from the data
        # itself (see compute_geometry / compute_inset_geometry). Since no such default can
        # be right for every combination of genome count and locus shape, all of them are
        # also exposed as flags, and their resolved values are printed back under the
        # COSMETICS section so they can be tuned.
        self.cosmetic_defaults = A('cosmetic_defaults') or {}

        self.main_graph_node_size = A('main_graph_node_size')
        self.main_graph_node_color = A('main_graph_node_color')
        self.main_graph_new_node_effect = A('main_graph_new_node_effect') or 'none'
        self.main_graph_nodes_fly_in = A('main_graph_nodes_fly_in') or False
        self.main_graph_node_edge_width = A('main_graph_node_edge_width')
        self.main_graph_node_edge_color = A('main_graph_node_edge_color') or '#000000'
        self.main_graph_edge_width = A('main_graph_edge_width')
        self.main_graph_edge_color = A('main_graph_edge_color') or '#3C3C3C'
        self.main_graph_edge_opacity = A('main_graph_edge_opacity') if A('main_graph_edge_opacity') is not None else 0.63

        self.inset_graph_level_height = A('inset_graph_level_height')
        self.inset_graph_node_size = A('inset_graph_node_size')
        self.inset_graph_node_color = A('inset_graph_node_color')
        self.inset_graph_new_node_effect = A('inset_graph_new_node_effect') or 'none'
        self.inset_graph_node_edge_width = A('inset_graph_node_edge_width')
        self.inset_graph_node_edge_color = A('inset_graph_node_edge_color') or '#000000'
        self.inset_graph_edge_width = A('inset_graph_edge_width')
        self.inset_graph_edge_color = A('inset_graph_edge_color') or '#3C3C3C'
        self.inset_graph_edge_opacity = A('inset_graph_edge_opacity') if A('inset_graph_edge_opacity') is not None else 0.55

        self.genome_track_line_width = A('genome_track_line_width')
        self.genome_track_line_color = A('genome_track_line_color')          # None -> anvi'o's own per-genome color
        self.genome_track_line_opacity = A('genome_track_line_opacity') if A('genome_track_line_opacity') is not None else 0.75
        self.genome_track_line_background = A('genome_track_line_background')  # None -> anvi'o's own track background

        self.fps = A('fps') or 30
        self.hold_first_seconds = A('hold_first_seconds') if A('hold_first_seconds') is not None else 1.0
        self.seconds_per_genome = A('seconds_per_genome') if A('seconds_per_genome') is not None else 0.45
        self.dissolve_seconds = A('dissolve_seconds') if A('dissolve_seconds') is not None else 0.2
        self.hold_last_seconds = A('hold_last_seconds') if A('hold_last_seconds') is not None else 1.0

        self.title_kicker = A('title_kicker') if A('title_kicker') is not None else 'The pangenome graph for'
        self.title = A('title')  # resolved from the pan-graph-db's project name if not given; see compute_frames
        self.subtitle = A('subtitle')
        self.inset_description = A('inset_description')  # resolved from --inset-flanks below if not given
        self.font_choice = A('font') or 'helvetica'
        self.background_color = A('background_color') or '#FFFFFF'
        self.keep_frames_dir = A('keep_frames_dir')
        self.export_stills = not A('no_export_stills')
        self.show_genome_tracks = not A('no_genome_tracks')
        self.dry_run = A('dry_run') or False

        self.check_dependencies()
        self.sanity_check()

        if self.inset_description is None and self.inset_flank_ids:
            left, right = self.inset_flank_ids
            self.inset_description = f"Subgraph between {left} and {right} detailed"

        # populated as processing proceeds
        self.frames = []                 # one dict per rendered genome-count state
        self.geom = {}                   # pinned geometry constants, computed once
        self.type_colors = {}
        self.total_genomes_rendered = None
        self.new_nodes_per_frame = []    # per frame, the nodes that were new to the graph in it

        # thread-local rather than one shared dict, see `get_font`
        self._fonts = threading.local()


    def check_dependencies(self):
        """Anvi'o itself doesn't need ffmpeg or Pillow; this program does. Fail loudly and
        early, rather than 80% of the way through a render."""

        missing = []

        if not utils.is_program_exists('ffmpeg', dont_raise=True):
            missing.append("`ffmpeg` was not found in your PATH. It is used to encode the final MP4 "
                           "(see https://ffmpeg.org for installation instructions, or `conda install ffmpeg`).")

        # asked of `importlib` rather than with a plain `import PIL`, since an import that only
        # exists to be tested for is an unused import as far as the linter is concerned
        if not importlib.util.find_spec('PIL'):
            missing.append("The `Pillow` Python package is not installed (`pip install Pillow`). It is used "
                           "to rasterize every frame.")

        if missing:
            raise ConfigError(f"This program needs a few things that are not strict anvi'o dependencies, "
                              f"and they seem to be missing from your environment: {' '.join(missing)}")


    def sanity_check(self):
        filesnpaths.is_file_exists(self.pan_graph_db_path)
        utils.is_pan_graph_db(self.pan_graph_db_path)
        filesnpaths.is_output_file_writable(self.output_file, ok_if_exists=False)

        if self.dry_run and not self.export_stills:
            raise ConfigError("--dry-run stops right after the first/last-frame stills are written, so there "
                              "would be nothing to look at with --no-export-stills also on. Drop one of the "
                              "two flags.")

        if self.max_num_genomes_to_render is not None and self.max_num_genomes_to_render < 1:
            raise ConfigError("--max-num-genomes-to-render must be 1 or greater.")

        if self.num_threads < 1:
            raise ConfigError("--num-threads is a number of frames to work on at the same time, so it has to "
                              "be at least 1.")

        num_cpus = os.cpu_count() or 1
        if self.num_threads > num_cpus:
            self.run.warning(f"You asked anvi'o to rasterize frames on {self.num_threads} threads, but this "
                             f"machine reports only {P('CPU', num_cpus)}. Anvi'o will go along with it, but "
                             f"past the number of CPUs the threads mostly get in one another's way rather "
                             f"than getting you your video any sooner.")

        if self.main_graph_nodes_fly_in and self.main_graph_new_node_effect != 'none':
            raise ConfigError(f"--main-graph-nodes-fly-in and --main-graph-new-node-effect "
                              f"('{self.main_graph_new_node_effect}') are two ways of marking the very same "
                              f"thing, the nodes a genome brought with it, and they would be drawing over one "
                              f"another: the effect fires where the node is going to be while the node itself "
                              f"is still on its way there. Please pick one of the two. Note that "
                              f"--inset-graph-new-node-effect is untouched by this, since nothing in the inset "
                              f"panel flies.")

        if self.inset_graph_level_height is not None and self.inset_graph_level_height <= 0:
            raise ConfigError("--inset-graph-level-height is a height in pixels, so it has to be greater "
                              "than zero.")

        if self.font_choice not in FONT_SETS:
            raise ConfigError(f"--font must be one of {', '.join(FONT_SETS.keys())}.")

        if self.inset_flanks_arg:
            flanks = [f.strip() for f in self.inset_flanks_arg.split(',')]
            if len(flanks) != 2:
                raise ConfigError(f"--inset-flanks takes exactly two comma-separated node ids (the two "
                                  f"backbone nodes that flank the locus you want magnified), but you gave "
                                  f"{len(flanks)}: {self.inset_flanks_arg}")
            self.inset_flank_ids = flanks
        else:
            self.inset_flank_ids = None

        try:
            self.canvas_w, self.canvas_h = (int(x) for x in self.resolution_arg.lower().split('x'))
        except Exception:
            raise ConfigError(f"--resolution must look like WIDTHxHEIGHT (e.g. '3840x2160'), not '{self.resolution_arg}'.")

        base, _ = os.path.splitext(self.output_file)
        self.still_paths = {
            'first-frame': (f"{base}-first-frame.svg", f"{base}-first-frame.png"),
            'last-frame': (f"{base}-last-frame.svg", f"{base}-last-frame.png"),
        }
        if self.export_stills:
            for svg_path, png_path in self.still_paths.values():
                filesnpaths.is_output_file_writable(svg_path, ok_if_exists=False)
                filesnpaths.is_output_file_writable(png_path, ok_if_exists=False)


    def px(self, v):
        """Scales a value authored against a 2160-tall frame to the actual canvas height."""

        return v * self.canvas_h / 2160.0


    def get_interactive_object(self):
        """A `PangraphInteractive` is a plain Python object, and it is the same class
        `anvi-display-pan-graph` hands to its bottle server. Here it is used with no server
        and no browser at all."""

        # imported lazily so that a missing anvio installation is reported by argparse's
        # normal import-time failure rather than by this program's own dependency check
        from anvio.interactive import PangraphInteractive

        interactive_args = argparse.Namespace(pan_graph_db=self.pan_graph_db_path, genomes_storage=None,
                                              skip_init_functions=True)

        d = PangraphInteractive(interactive_args, run=terminal.Run(verbose=False), progress=terminal.Progress(verbose=False))

        # mirrors what the interface's initial page load does (`initial_pangraph_json_data`
        # in bottleroutes.py) before the first `rerun_state` call
        d.load_state('default', 'default')

        return d


    def get_genome_names_from_parameter(self, value, flag):
        """Genome names from a parameter that accepts either a comma-separated list of names
        or a path to a file with one name per line."""

        if os.path.exists(value):
            with open(value) as f:
                names = [line.strip() for line in f if line.strip()]
        else:
            names = [g.strip() for g in value.split(',') if g.strip()]

        if not names:
            raise ConfigError(f"{flag} did not resolve to any genome names.")

        return names


    def resolve_genomes(self, d):
        """The genomes to render, in the order they will appear in the animation.

        Every genome in the pan-graph-db is rendered by default. `--genomes-of-interest`
        narrows that set down and fixes the order of appearance, and `--exclude-genomes`
        drops names from whatever set is left."""

        all_genomes = list(d.genome_names)

        if self.genomes_of_interest_arg:
            genomes = self.get_genome_names_from_parameter(self.genomes_of_interest_arg, '--genomes-of-interest')

            unknown = sorted(set(genomes) - set(all_genomes))
            if unknown:
                raise ConfigError(f"--genomes-of-interest names {len(unknown)} genome(s) that are not in this "
                                  f"pan-graph-db: {', '.join(unknown[:5])}"
                                  f"{' (and more)' if len(unknown) > 5 else ''}.")

            repeated = sorted({g for g in genomes if genomes.count(g) > 1})
            if repeated:
                raise ConfigError(f"--genomes-of-interest lists these genome(s) more than once: "
                                  f"{', '.join(repeated)}.")
        else:
            genomes = all_genomes

        if self.exclude_genomes_arg:
            excluded = set(self.get_genome_names_from_parameter(self.exclude_genomes_arg, '--exclude-genomes'))

            unknown = sorted(excluded - set(all_genomes))
            if unknown:
                raise ConfigError(f"--exclude-genomes names {len(unknown)} genome(s) that are not in this "
                                  f"pan-graph-db: {', '.join(unknown[:5])}"
                                  f"{' (and more)' if len(unknown) > 5 else ''}.")

            genomes = [g for g in genomes if g not in excluded]

            if not genomes:
                raise ConfigError("--exclude-genomes removed every single genome that was left to render, so "
                                  "there is nothing to draw. Please exclude fewer genomes.")

        return genomes


    def compute_frames(self):
        """The heart of the whole program: for genome counts 1 through N, ask anvi'o to
        recompute the pangenome graph restricted to the first that-many genomes, and keep
        the resulting nodes/edges/layout. Every one of those steps is a cheap
        subgraph-and-relayout operation (see `PanGraphSuperclass.init_pangenome_graph` in
        anvio/dbops.py) rather than a rerun of the gene-cluster fusion engine, which is what
        makes it practical to do this once per genome."""

        d = self.get_interactive_object()
        genomes = self.resolve_genomes(d)

        if self.max_num_genomes_to_render and self.max_num_genomes_to_render > len(genomes):
            self.run.warning(f"You asked for {self.max_num_genomes_to_render} genomes to be rendered, but "
                             f"there are only {len(genomes)} of them available (which is either every genome "
                             f"in the pan-graph-db, or whatever your --genomes-of-interest and "
                             f"--exclude-genomes selections left behind), so anvi'o will render all of them.")

        num_genomes_to_render = min(self.max_num_genomes_to_render or len(genomes), len(genomes))
        genomes = genomes[:num_genomes_to_render]

        self.run.warning(None, header="COMPUTING PANGENOME GRAPH GROWTH FRAMES", lc="green")
        self.run.info("Pan graph db", self.pan_graph_db_path)
        self.run.info("Genomes available", len(d.genome_names))
        self.run.info("Genomes rendered", f"{num_genomes_to_render} (frames go from 1 genome to {num_genomes_to_render})")
        self.run.info("Component", self.component)

        self.progress.new("Recomputing graph layout per genome count", progress_total_items=num_genomes_to_render)

        # a frame that could not show the component that was asked for is worth mentioning, but
        # NOT from inside the loop: a warning printed while the progress bar is live would be
        # written straight over it, and there can be one of these per frame. They are collected
        # here and reported as one block once the progress bar is done with.
        substitutions = []

        last_used_component = None
        for n in range(1, num_genomes_to_render + 1):
            self.progress.update(f"{n}/{num_genomes_to_render} genomes")

            genomes_in_frame = genomes[:n]
            d.rerun_state(gene_cluster_grouping_threshold=-1, groupcompress=1.0, max_edge_length_filter=-1,
                          component=self.component, genomes=genomes_in_frame)

            data = d.get_json()

            used_component = data['meta'].get('component')
            if last_used_component is not None and used_component != last_used_component:
                substitutions.append((n, used_component))
            last_used_component = used_component

            self.frames.append({'n': n, 'genomes': genomes_in_frame, 'data': data})
            self.progress.increment()

        self.progress.end()

        if substitutions:
            where = ', '.join(f"'{used_component}' from {n} genome(s) on" for n, used_component in substitutions)
            self.run.warning(f"Component '{self.component}' is not what anvi'o ended up drawing in every single "
                             f"frame. A component that isn't there yet (or that has changed shape) at a given "
                             f"genome count leaves anvi'o showing another one for that frame instead, which is "
                             f"expected for the early frames of a growing pangenome. Here is what was drawn "
                             f"instead, and from which frame onwards: {where}.")

        self.type_colors = self.frames[-1]['data']['states']['nodes']['type_colors']
        self.total_genomes_rendered = self.frames[-1]['n']

        # which nodes were new in which frame. Worked out once, here, because `new_node_ids`
        # compares a frame against the one before it and the answer never changes afterwards. The
        # sets are disjoint by construction: a node is new in exactly one frame, ever.
        self.new_nodes_per_frame = [self.new_node_ids(i) for i in range(len(self.frames))]

        if self.title is None:
            self.title = self.frames[-1]['data']['meta'].get('project_name') or ''

        if self.inset_flank_ids:
            self.validate_inset_flanks()


    def validate_inset_flanks(self):
        left, right = self.inset_flank_ids
        for frame in self.frames:
            nodes = frame['data']['nodes']
            missing = [node_id for node_id in (left, right) if node_id not in nodes]
            if missing:
                raise ConfigError(f"--inset-flanks node(s) {', '.join(missing)} are not in the pangenome graph "
                                  f"at {frame['n']} genome(s). This program needs both flanking nodes to be "
                                  f"present in EVERY frame you're rendering, so pick two nodes that are part "
                                  f"of the conserved backbone, and are therefore there from the very first "
                                  f"genome onward. You can find candidate node/gene-cluster ids by looking at "
                                  f"your pan-graph-db in `anvi-display-pan-graph`.")


    def compute_geometry(self):
        """Pin one set of scale constants derived from the LAST (largest) frame, and reuse
        them for every frame. That is what makes the growth read as the graph expanding
        outward against a fixed frame of reference, rather than the whole picture being
        rescaled to fill the canvas every time.

        The overall arrangement is a square graph panel on the right, and a column of text
        on the left that optionally also holds a magnified inset panel."""

        final_nodes = self.frames[-1]['data']['nodes']
        final_x_max = max(d['position'][0] for d in final_nodes.values())

        has_inset = self.inset_flank_ids is not None

        # The whole right-side panel (tracks + graph together) always fills as much of the
        # canvas as it can, and is NOT what shrinks as genomes accumulate. Only the graph's
        # own radial reach beyond the tracks (see `graph_height_fraction` below) gets
        # smaller.
        #
        # The panel is a SQUARE, so its size is whichever is tighter: the width left over
        # once the text/inset column is reserved, or the canvas height. This is computed
        # from the actual --resolution and column width every time, rather than being a
        # fixed constant tuned for one particular canvas size, so the panel always claims
        # the most space it can without overlapping the column or running off the canvas.
        # The column is the SAME whether or not there is an inset to put in it. Everything in
        # it (the title block, the counter, the progress bar) therefore lands in exactly the
        # same place either way, and turning the inset off subtracts the description and the
        # panel without shifting or resizing anything else.
        gap_to_column_px = self.px(40)
        edge_margin_px = self.px(20)
        col_x, col_w = self.px(140), self.px(1580)
        available_width = self.canvas_w - col_x - col_w - gap_to_column_px - edge_margin_px

        graph_size = min(available_width, self.canvas_h - 2 * edge_margin_px)
        graph_x = self.canvas_w - graph_size - edge_margin_px
        graph_y = (self.canvas_h - graph_size) / 2.0

        if graph_size < self.px(200):
            self.run.warning("At this --resolution, the text column leaves very little room for the graph "
                             "panel, so it may look extremely cramped.")

        main_center = (graph_x + graph_size / 2.0, graph_y + graph_size / 2.0)

        # The radius is split into three bands: an empty hole, a genome-tracks band with one
        # FIXED-height ring per genome (fixed against the FINAL genome count, so an existing
        # ring never resizes when a new one is added, it just gains a new neighbor), and the
        # backbone/growth zone right outside it, separated by a small gap that is also fixed.
        # The backbone radius depends on genome count and on NOTHING else, and in particular
        # not on how long that frame's backbone happens to be, so it hugs the track stack by
        # a constant amount at every frame, and grows by exactly one track's worth each time
        # a genome is added.
        #
        # The height of the tracks band has no knob of its own. Everything needed to fill the
        # panel EXACTLY, with no leftover whitespace and nothing clipped, is already known:
        # the panel's target radius, the hole, and how big the graph's own reach should be
        # relative to the tracks (--graph-height-fraction, g). So the one unknown, each
        # track's height, is solved for:
        #
        #   hole + tracks_total + g * tracks_total = target_outer_radius_px
        #   tracks_total = (target_outer_radius_px - hole) / (1 + g)
        #
        # and tracks_total is divided evenly across every genome being rendered. The graph's
        # own reach (g * tracks_total) is then split into a small fixed hug-gap plus however
        # much spire-climb is left, sized from the FINAL frame's REAL tallest spire, so that a
        # shallow locus and a tall one both land exactly on the same target radius.
        target_outer_radius_px = self.outer_radius_fraction * (graph_size / 2.0)
        hole_radius_px = self.inner_radius_fraction * target_outer_radius_px
        g = self.graph_height_fraction

        tracks_total_px = max(0.0, target_outer_radius_px - hole_radius_px) / (1.0 + g)
        fixed_track_height_px = tracks_total_px / self.total_genomes_rendered
        graph_reach_px = g * tracks_total_px
        backbone_gap_px = self.backbone_gap_px if self.backbone_gap_px is not None else fixed_track_height_px * 0.5

        # anvi'o's own radial pangenome display starts its sweep at 6 o'clock and goes
        # counter-clockwise; `theta()` below reproduces that (see its docstring), so only the
        # SPAN of the sweep is taken from the pan-graph-db's state, not its absolute angles.
        drawing_state = self.frames[-1]['data']['states']['drawing']
        angular_span = abs(math.radians(drawing_state.get('end_angle', 270) - drawing_state.get('start_angle', 0)))

        final_y_max = self.graph_y_max(self.frames[-1]['data'])

        if self.track_height_px is not None:
            disty_px = self.track_height_px
        elif final_y_max > 0:
            disty_px = max(0.0, graph_reach_px - backbone_gap_px) / final_y_max
        else:
            disty_px = 0.0

        self.geom = {
            'has_inset': has_inset,
            'col_x': col_x, 'col_w': col_w,
            'panel_box': self.inset_panel_box(col_x, col_w),
            'disty_px': disty_px,
            'hole_radius_px': hole_radius_px,
            'fixed_track_height_px': fixed_track_height_px,
            'genome_track_line_width': self.genome_track_line_width if self.genome_track_line_width is not None else max(0.6, fixed_track_height_px * 0.035),
            'backbone_gap_px': backbone_gap_px,
            'angular_span': angular_span,
            'rotation_offset_rad': math.radians(self.rotation_offset_deg),
            'main_center': main_center,
        }

        # Node/edge ink is sized from how tightly the FINAL frame's backbone (the longest
        # one) actually packs around its own hugging radius, which is to say from the real
        # geometry that hugging produces. These are all DEFAULTS, and every one of them is
        # directly overridable (see the COSMETICS section printed before rendering starts),
        # since no formula can look right for every combination of genome count and locus
        # shape.
        final_radius = self.backbone_radius(self.total_genomes_rendered)
        distx_px = final_radius * angular_span / final_x_max if final_x_max else 1.0
        self.geom['distx_px'] = distx_px

        node_size = self.main_graph_node_size if self.main_graph_node_size is not None else max(1.0, distx_px * 0.4)
        self.geom['main_graph_node_size'] = node_size
        self.geom['main_graph_node_edge_width'] = self.main_graph_node_edge_width if self.main_graph_node_edge_width is not None else max(0.5, node_size / 3.0)
        self.geom['main_graph_node_edge_color'] = self.hex_to_rgb(self.main_graph_node_edge_color)
        self.geom['main_graph_edge_width'] = self.main_graph_edge_width if self.main_graph_edge_width is not None else max(0.5, distx_px * 0.35)
        self.geom['main_graph_edge_color'] = self.hex_to_rgb(self.main_graph_edge_color)
        self.geom['main_graph_edge_opacity'] = self.main_graph_edge_opacity

        if self.track_height_px is not None:
            final_spire_reach = final_radius + disty_px * final_y_max
            if final_spire_reach > target_outer_radius_px:
                self.run.warning(f"With this --track-height-px, the finished graph's tallest spire reaches "
                                 f"about {int(final_spire_reach)}px from center, past the "
                                 f"{int(target_outer_radius_px)}px --outer-radius-fraction reference (of a "
                                 f"{int(graph_size / 2)}px graph-panel half-size), so it may run close to, or "
                                 f"past, the edge of the graph panel.")

        self.compute_counter_text_geometry()

        if has_inset:
            self.compute_inset_geometry()

        self.compute_counter_block_layout()


    def inset_panel_box(self, col_x, col_w):
        """Where the inset panel sits: as tall as --inset-aspect makes it, centered in the band
        of the column reserved for it.

        This depends on nothing but the column and that aspect, so it can be worked out whether
        or not there is actually an inset to draw. That is what lets the counter block above it
        stack against the same edge either way."""

        panel_h = int(round(col_w / self.inset_aspect))
        band_top, band_bottom = self.px(PANEL_TOP), self.px(PANEL_BOTTOM)
        panel_y = band_top + max(0, ((band_bottom - band_top) - panel_h) / 2.0)

        return (col_x, panel_y, col_w, panel_h)


    def compute_inset_geometry(self):
        """Pins the inset's ZOOM SCALE (and the panel box it draws into) from the FINAL
        frame only. It deliberately does NOT pin which backbone-position numbers count as
        'inside the window', because those numbers come from a fresh layout every frame (see
        `compute_geometry`'s docstring), so a node's x position in an early frame is not
        comparable to its x position in the final frame. Each frame re-derives its own
        window from its own two flank positions (see `add_inset_shapes`), and only the
        resulting SCALE (pixels per backbone unit, and per y unit) is held fixed across
        frames."""

        left, right = self.inset_flank_ids
        final_nodes = self.frames[-1]['data']['nodes']
        x_left = final_nodes[left]['position'][0]
        x_right = final_nodes[right]['position'][0]
        x_lo, x_hi = sorted([x_left, x_right])

        window_ids = {nid for nid, d in final_nodes.items() if x_lo <= d['position'][0] <= x_hi}
        if not window_ids:
            raise ConfigError("The two --inset-flanks nodes don't bracket any nodes in the final frame, so "
                              "please double check the two node ids.")

        # y=0 is the universal flat-backbone baseline in every frame by construction (it is
        # never negative), so the floor to anchor against is always 0, rather than something
        # to measure from whichever nodes happen to be in this one frame's window.
        y_hi = self.inset_window_y_max(self.frames[-1]['data'], window_ids)

        panel_box = self.geom['panel_box']

        margin = 0.06 * min(panel_box[2], panel_box[3])
        available_w = panel_box[2] - 2 * margin
        available_h = panel_box[3] - 2 * margin

        # The two axes of the inset are scaled INDEPENDENTLY, and that is the whole reason the
        # panel magnifies anything. A locus is usually far wider than it is tall, so a single
        # scale fitted to both would be pinned by the width, and the handful of y levels the
        # subgraph does have would end up stacked a few pixels apart: unreadable, and exactly
        # what the panel exists to avoid. Giving the levels the panel's full height instead
        # spreads them as far apart as the box allows.
        scale = available_w / max(1e-6, x_hi - x_lo)
        level_height = self.inset_graph_level_height
        if level_height is None:
            level_height = available_h / y_hi if y_hi > 0 else 0.0

        # a node's on-screen footprint should track how zoomed-in the inset actually is
        # rather than just the output resolution, otherwise a tight zoom looks sparse and a
        # wide one looks clogged. The TIGHTER of the two axes sets it, since that is the one
        # that decides when neighbouring nodes start to touch. Still only a default (see
        # --inset-graph-node-size).
        ink_scale = min(scale, level_height) if level_height > 0 else scale
        node_size = self.inset_graph_node_size if self.inset_graph_node_size is not None else max(4.0, ink_scale * 0.42)

        self.geom['inset'] = {
            'margin': margin,
            'available_w': available_w,
            'scale': scale,
            'level_height': level_height,
            'node_r': node_size,
            'node_edge_width': self.inset_graph_node_edge_width if self.inset_graph_node_edge_width is not None else max(0.5, node_size / 6.0),
            'node_edge_color': self.hex_to_rgb(self.inset_graph_node_edge_color),
            'edge_w': self.inset_graph_edge_width if self.inset_graph_edge_width is not None else max(1.0, ink_scale * 0.08),
            'edge_color': self.hex_to_rgb(self.inset_graph_edge_color),
            'edge_opacity': self.inset_graph_edge_opacity,
        }


    def tracks_outer_radius(self, n_genomes):
        """Outer edge of the genome-tracks stack for a frame with `n_genomes` tracks drawn.
        Each track has the SAME fixed height (pinned against the final genome count, see
        `compute_geometry`), so this is simply that stack's current height. It grows one
        fixed increment at a time as genomes are added, never resizing an existing ring."""

        return self.geom['hole_radius_px'] + n_genomes * self.geom['fixed_track_height_px']


    def backbone_radius(self, n_genomes):
        """Radius the backbone ring (y=0) sits at: right outside (hugging) the current
        frame's own track stack, plus a small FIXED gap. It is deliberately independent of
        how long that frame's backbone happens to be, so growth across frames comes entirely
        from genome count, and the gap between the tracks and the backbone stays constant."""

        return self.tracks_outer_radius(n_genomes) + self.geom['backbone_gap_px']


    def theta(self, x, x_max):
        """Screen angle for backbone position `x` (of `x_max`), in the same convention
        anvi'o's own radial pangenome display uses: the sweep starts at 6 o'clock and runs
        COUNTER-clockwise. In screen coordinates (y grows downward), 6 o'clock is +90 degrees
        and counter-clockwise means the angle DECREASES as x grows, which is the opposite of
        the usual "0 degrees is 3 o'clock, increasing is clockwise" convention."""

        g = self.geom
        progress = (x / x_max) * g['angular_span'] if x_max else 0.0
        return math.pi / 2 - progress + g['rotation_offset_rad']


    def project_circular(self, x, y, x_max, n_genomes):
        g = self.geom
        theta = self.theta(x, x_max)
        r = self.backbone_radius(n_genomes) + y * g['disty_px']
        cx, cy = g['main_center']
        return (cx + r * math.cos(theta), cy + r * math.sin(theta))


    def inset_x_scale(self, x_lo, x_hi):
        """Pixels per backbone step inside the inset panel, for a frame whose window runs from
        `x_lo` to `x_hi`.

        By default this is the scale pinned to the FINAL frame, which means a subgraph that
        grows as genomes arrive starts out small and nested against the left edge of the panel,
        then expands rightwards until the last genome fills the panel exactly. That reads as
        the locus being discovered a piece at a time.

        With --inset-graph-nodes-cover-window-dynamically it is solved per frame from that
        frame's OWN flank span instead, so the window spans the panel's full width in every
        frame. What the eye follows is then structure appearing between two fixed flanks,
        rather than the whole drawing sliding rightwards. Node and edge ink stays pinned either
        way, so only the spacing between nodes moves."""

        inset = self.geom['inset']

        if not self.inset_nodes_cover_window_dynamically:
            return inset['scale']

        return inset['available_w'] / max(1e-6, x_hi - x_lo)


    def project_inset(self, x, y, x_lo, x_hi):
        """`x_lo`/`x_hi` are THIS FRAME's own flank positions (see `add_inset_shapes`), and
        never the final frame's, since positions aren't comparable across frames. `y` is a
        track/lane index that grows AWAY from the flat conserved backbone (think of it as
        height above a floor): on screen that has to mean UP, and the floor has to sit near
        the panel's bottom so growth fills the panel instead of spreading evenly around a
        centered midline."""

        inset = self.geom['inset']
        ox, oy, _panel_w, panel_h = self.geom['panel_box']
        margin = inset['margin']

        px_ = margin + (x - x_lo) * self.inset_x_scale(x_lo, x_hi)
        py_ = (panel_h - margin) - y * inset['level_height']
        return (ox + px_, oy + py_)


    def get_font(self, size_px, role):
        """A face of the chosen typeface at a given size.

        The cache behind this is THREAD-LOCAL, and deliberately so. Pillow's `FreeTypeFont` wraps
        a single FreeType face, and FreeType promises nothing about one face being drawn with from
        two threads at once, so the frame workers (see `run_in_parallel`) each get their own copy
        of every face they touch rather than taking turns on one. A whole video only ever asks for
        a handful of (size, role) combinations, so the duplication costs nothing worth measuring."""

        cache = getattr(self._fonts, 'cache', None)
        if cache is None:
            cache = self._fonts.cache = {}

        key = (int(round(size_px)), role)
        if key not in cache:
            from PIL import ImageFont
            path, faces, _family = FONT_SETS[self.font_choice]
            try:
                cache[key] = ImageFont.truetype(path, max(1, key[0]), index=faces.get(role, faces['regular']))
            except Exception:
                cache[key] = self._fallback_font(key[0])
        return cache[key]


    @staticmethod
    def _fallback_font(size_px):
        from PIL import ImageFont
        for font_path in ["/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf", "Arial.ttf"]:
            try:
                return ImageFont.truetype(font_path, max(1, size_px))
            except Exception:
                continue
        return ImageFont.load_default()


    # ------------------------------------------------------------------
    # Building a 'scene': a backend-agnostic list of shapes and text ops
    # in canvas-pixel coordinates. `rasterize_scene` (PIL) and `scene_to_svg`
    # (plain SVG) both consume the same scene, which is what guarantees an
    # exported still frame matches its corresponding video frame exactly.
    # ------------------------------------------------------------------

    def new_node_ids(self, frame_idx):
        """The nodes that arrived with this frame's genome: whatever this frame's graph holds
        that the frame before it did not.

        Node ids stay identical from one genome subset to the next (see `init_pangenome_graph`
        in anvio/dbops.py), which is the only reason this comparison means anything. Their
        POSITIONS do move, since every frame is laid out afresh, so a node is always drawn
        where THIS frame puts it.

        The first frame has none. Every node in it is new only in the trivial sense that there
        was nothing before it, and marking a whole graph at once says nothing about what any
        one genome brought to it."""

        if frame_idx == 0:
            return set()

        return (set(self.frames[frame_idx]['data']['nodes'])
                - set(self.frames[frame_idx - 1]['data']['nodes']))


    def genes_added_node_ids(self, frame_idx):
        """The NON-CORE nodes the genome arriving with this frame put a gene into.

        This is a deliberately wider net than `new_node_ids`, and it is what the inset panel
        marks. A node does not have to be new to qualify, only to have gained something: a
        genome dropping a gene into a synteny gene cluster that several other genomes already
        occupy has changed that cluster, and at the inset's magnification that is worth seeing
        even though the node was already on screen. The main panel deliberately does NOT do
        this, because at whole-graph scale hundreds of marks per genome would be noise.

        Core nodes are left out on purpose. 'core' means every genome currently in the graph has
        a gene there (see `recompute_node_types` in anvio/dbops.py), so every genome would light
        up every core node and the effect would say nothing about any particular one of them.

        A brand new node always qualifies, since the arriving genome is by definition why it
        appeared and a node new to this frame cannot be core, so this strictly contains
        `new_node_ids`."""

        if frame_idx == 0:
            return set()

        genome = self.frames[frame_idx]['genomes'][-1]

        return {node_id for node_id, d in self.frames[frame_idx]['data']['nodes'].items()
                if d.get('type') != 'core' and genome in d.get('gene_calls', {})}


    def enabled_new_node_effects(self):
        """The effects the two panels asked for, as instances. Empty when neither wants one, in
        which case a genome's turn on screen stays a single image repeated and the video costs
        exactly what it always did."""

        return [NEW_NODE_EFFECTS[name]
                for name in (self.main_graph_new_node_effect, self.inset_graph_new_node_effect)
                if name in NEW_NODE_EFFECTS]


    def overlay_duration_fractions(self):
        """Every span, as a share of --seconds-per-genome, that something animated over a genome's
        arrival asks for: the two panels' new-node effects, and the outer ring's flight. Empty
        when nothing at all is animating, in which case a genome's turn stays one still image."""

        fractions = [effect.duration_fraction for effect in self.enabled_new_node_effects()]

        if self.main_graph_nodes_fly_in:
            fractions.append(MAIN_GRAPH_FLY_IN_FRACTION)

        return fractions


    def overlay_frames(self):
        """How many frames the arrival overlay plays for. Everything animating over a genome's
        turn shares ONE timeline, so whichever of them asks for the longest span sets it."""

        fractions = self.overlay_duration_fractions()

        if not fractions:
            return 0

        return max(1, int(round(max(fractions) * self.seconds_per_genome * self.fps)))


    def count_video_frames(self, effect_frames):
        """How many frames `build_video`'s timeline will write, counted the same way it writes
        them. Lets the run say up front how long the video runs and how much rasterizing it is
        in for, and gives the progress bar a real total to work against."""

        dissolve_frames, arrival_frames, turn_frames = self.genome_turn_frames(effect_frames)

        # the first genome gets no effect (see `new_node_ids`), so its hold is untouched
        total = max(1, int(self.hold_first_seconds * self.fps))
        total += (len(self.frames) - 1) * (arrival_frames + max(1, turn_frames - arrival_frames))
        total += dissolve_frames + max(1, int(self.hold_last_seconds * self.fps))

        return total


    def genome_turn_frames(self, effect_frames):
        """`(dissolve_frames, arrival_frames, turn_frames)` for one genome's turn on screen.

        A turn lasts --dissolve-seconds of fading in plus --seconds-per-genome on screen, and
        that is `turn_frames`. `arrival_frames` is the front of it that has to be rasterized one
        frame at a time, because either the graph is still fading in or the effect is still
        running; whatever is left of the turn after that is one still image repeated.

        The effect and the fade OVERLAP rather than queue, which is why this is a max and not a
        sum: an effect that waited for the fade to finish put a visible pause between a genome
        landing and anything happening about it. Overlapping them also means an effect that fits
        inside a turn does not lengthen the video at all."""

        dissolve_frames = max(0, int(round(self.dissolve_seconds * self.fps)))
        turn_frames = dissolve_frames + int(self.seconds_per_genome * self.fps)

        return dissolve_frames, max(effect_frames, dissolve_frames), turn_frames


    def arrival_overlay_shapes(self, frame_idx, progress):
        """Shapes marking what the genome arriving with this frame changed, at one point in the
        overlay's span. These go OVER a frame's static scene rather than into it, so the same
        scene can be re-rendered at successive points of the overlay without rebuilding
        everything underneath it.

        The two panels mark DIFFERENT things on purpose. The main panel takes `new_node_ids`,
        nodes that were not there before, since at whole-graph scale anything wider is noise.
        The inset takes `genes_added_node_ids`, every non-core node the genome put a gene into,
        new or not, since at that magnification a gene joining an existing cluster is exactly
        what one wants to see. See those two methods for the reasoning."""

        shapes = []
        frame = self.frames[frame_idx]
        nodes = frame['data']['nodes']
        x_max = max((d['position'][0] for d in nodes.values()), default=1)

        def fill_for(node, override):
            if override:
                return self.hex_to_rgb(override)
            return self.hex_to_rgb(self.type_colors.get(node.get('type'), '#999999'))

        main_effect = NEW_NODE_EFFECTS.get(self.main_graph_new_node_effect)
        if main_effect:
            n_genomes = len(frame['genomes'])
            for node_id in self.new_node_ids(frame_idx):
                d = nodes[node_id]
                x, y = self.project_circular(d['position'][0], d['position'][1], x_max, n_genomes)
                shapes.extend(main_effect.shapes(x, y, self.geom['main_graph_node_size'],
                                         fill_for(d, self.main_graph_node_color),
                                         self.geom['main_graph_node_edge_color'],
                                         self.geom['main_graph_node_edge_width'], progress))

        inset_effect = NEW_NODE_EFFECTS.get(self.inset_graph_new_node_effect)
        if inset_effect and self.geom.get('has_inset'):
            inset = self.geom['inset']
            x_lo, x_hi, window_ids = self.inset_window(frame_idx)
            for node_id in self.genes_added_node_ids(frame_idx) & window_ids:
                d = nodes[node_id]
                x, y = self.project_inset(*d['position'], x_lo, x_hi)
                shapes.extend(inset_effect.shapes(x, y, inset['node_r'],
                                          fill_for(d, self.inset_graph_node_color),
                                          inset['node_edge_color'], inset['node_edge_width'], progress))

        # and this genome's own new nodes, still on their way to where the graph has put them
        if self.main_graph_nodes_fly_in:
            shapes.extend(self.fly_in_shapes(frame_idx, progress))

        return shapes


    def fly_in_shapes(self, frame_idx, progress):
        """The nodes this frame's genome brought, part-way through their flight in from off-canvas.

        Each one comes in from its own fixed bearing, oversized and shrinking as it goes, easing to
        a stop at exactly its true size in the place the graph has for it. The size is what makes
        this legible at all: nodes in a dense pangenome resolve to a pixel or two across, and a
        single pixel crossing the frame cannot be followed however opaque it is. Arriving large and
        settling also reads as depth, something approaching from far off, and it self-corrects,
        since the size a node lands at is the only one that is real.

        The bearing comes from the node's id rather than being drawn at random: a node that picked
        a new direction on every frame of its own flight would flicker rather than fly, and two
        runs of the same video would not match each other.

        Whether it also fades up as it travels is MAIN_GRAPH_FLY_IN_FADE. A fade reads well on a
        big node and badly on a small one, where it just makes something already hard to see nearly
        invisible for most of its journey, so it is off. Nothing pops into view either way: a node
        sets off further from the centre than the furthest corner of the canvas, so the whole of
        its first stretch happens off screen.

        What lands is the node as `build_scene` would have drawn it — same radius, same type color,
        same outline — so there is nothing to see at the end of the flight but the graph the frame
        was always going to hold.

        Edges are deliberately left alone, and are drawn in full from the first frame of the
        flight. An edge already reaching for a node still on its way in reads as the graph holding
        a place open for it, where fading every edge that touches a new node would instead have the
        whole neighbourhood flicker each time a genome arrives."""

        frame = self.frames[frame_idx]
        nodes = frame['data']['nodes']
        x_max = max((d['position'][0] for d in nodes.values()), default=1)
        n_genomes = len(frame['genomes'])

        cx, cy = self.geom['main_center']
        start_radius = math.hypot(self.canvas_w, self.canvas_h) * MAIN_GRAPH_FLY_IN_START_DISTANCE
        travelled = NewNodeEffect.ease_out(progress)

        # size follows the SAME eased progress the position does, so a node reaches its true
        # footprint at the exact moment it reaches its true place, and the two never disagree
        size_scale = 1.0 + (MAIN_GRAPH_FLY_IN_START_SIZE - 1.0) * (1.0 - travelled)

        radius = self.geom['main_graph_node_size']
        outline = self.geom['main_graph_node_edge_color']
        outline_width = self.geom['main_graph_node_edge_width']
        override = self.hex_to_rgb(self.main_graph_node_color) if self.main_graph_node_color else None

        shapes = []

        for node_id in self.new_nodes_per_frame[frame_idx]:
            d = nodes.get(node_id)
            if d is None:
                continue

            land_x, land_y = self.project_circular(d['position'][0], d['position'][1], x_max, n_genomes)

            bearing = int(hashlib.md5(node_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF * 2 * math.pi
            start_x = cx + start_radius * math.cos(bearing)
            start_y = cy + start_radius * math.sin(bearing)

            color = override or self.hex_to_rgb(self.type_colors.get(d.get('type'), '#999999'))

            shapes.append(('fading_circle', start_x + (land_x - start_x) * travelled,
                           start_y + (land_y - start_y) * travelled,
                           radius * size_scale, color, outline, outline_width * size_scale,
                           progress if MAIN_GRAPH_FLY_IN_FADE else 1.0))

        return shapes


    def build_scene(self, frame_idx, name_the_genome_added=True, arrivals_in_flight=False):
        """This frame's whole picture, as shapes and text.

        `arrivals_in_flight` leaves this frame's own new NODES out, which is what the arrival
        overlay wants: while they are still on their way in they must not also be sitting at their
        destinations underneath it. The frame is otherwise identical, so the held part of a
        genome's turn and the flight share everything but those few nodes."""

        frame = self.frames[frame_idx]
        nodes = frame['data']['nodes']
        x_max = max(d['position'][0] for d in nodes.values()) if nodes else 1
        n_genomes = len(frame['genomes'])

        shapes, texts = [], []

        if self.show_genome_tracks:
            self.add_genome_track_shapes(shapes, frame_idx, x_max)

        if self.geom.get('has_inset'):
            self.add_inset_marker_shapes(shapes, frame_idx, x_max, n_genomes)

        backbone_radius = self.backbone_radius(n_genomes)
        projector = self.radial_projector(x_max, lambda y: backbone_radius + y * self.geom['disty_px'])

        for edge in self.active_edges(frame['data']):
            u, v = edge['source'], edge['target']
            if u not in nodes or v not in nodes:
                continue
            chain = self.edge_chain(nodes[u]['position'], nodes[v]['position'], edge.get('route'))
            pixels = self.project_chain(chain, *projector)
            if len(pixels) >= 2:
                shapes.append(('polyline', pixels, self.geom['main_graph_edge_color'], self.geom['main_graph_edge_width'], self.geom['main_graph_edge_opacity']))

        node_color = self.hex_to_rgb(self.main_graph_node_color) if self.main_graph_node_color else None
        in_flight = self.new_nodes_per_frame[frame_idx] if arrivals_in_flight else set()

        for node_id, d in nodes.items():
            if node_id in in_flight:
                continue
            x, y = self.project_circular(d['position'][0], d['position'][1], x_max, n_genomes)
            color = node_color or self.hex_to_rgb(self.type_colors.get(d.get('type'), '#999999'))
            r = self.geom['main_graph_node_size']
            shapes.append(('circle', x, y, r, color, self.geom['main_graph_node_edge_color'], self.geom['main_graph_node_edge_width']))

        if self.geom.get('has_inset'):
            self.add_inset_shapes(shapes, frame_idx)

        self.add_text_ops(texts, shapes, frame['n'], self.total_genomes_rendered,
                          frame['genomes'][-1] if name_the_genome_added else None)

        return {'shapes': shapes, 'texts': texts}


    @staticmethod
    def wedge_polygon_points(cx, cy, r0, r1, theta0, theta1, n=16):
        outer = [(cx + r1 * math.cos(theta0 + (theta1 - theta0) * i / (n - 1)),
                 cy + r1 * math.sin(theta0 + (theta1 - theta0) * i / (n - 1))) for i in range(n)]
        inner = [(cx + r0 * math.cos(theta1 - (theta1 - theta0) * i / (n - 1)),
                 cy + r0 * math.sin(theta1 - (theta1 - theta0) * i / (n - 1))) for i in range(n)]
        return outer + inner


    def project_at_radius(self, x, r, x_max):
        """Like `project_circular`, but at a caller-chosen radius instead of one derived
        from the backbone/growth-zone math. Used for the genome tracks, whose radius comes
        from their own ring rather than from `backbone_radius`."""

        theta = self.theta(x, x_max)
        cx, cy = self.geom['main_center']
        return (cx + r * math.cos(theta), cy + r * math.sin(theta))


    @staticmethod
    def active_edges(data):
        """The edges of a frame that actually get drawn. `get_json` ships every edge of the
        active component, including the ones a max-edge-length filter has switched off, and
        anvi'o's interface skips those at draw time (`edge['active'] == true`) rather than
        dropping them from the data."""

        return [edge for edge in data['edges'].values() if edge.get('active', True)]


    def graph_y_max(self, data):
        """The largest y (track/spire level) anything in a frame reaches. Edge ROUTE waypoints
        count towards it and not just nodes, since a route is free to climb above every node it
        passes over, and a y scale blind to that would let those waypoints overshoot whatever
        they were scaled to fit. This is the same quantity the interface calls `global_y`."""

        y_max = max((d['position'][1] for d in data['nodes'].values()), default=0)

        for edge in self.active_edges(data):
            for _x, y in (edge.get('route') or []):
                y_max = max(y_max, y)

        return y_max


    def inset_window_y_max(self, data, window_ids):
        """The tallest y anything inside the magnified window reaches. Edge ROUTE waypoints
        count towards it and not just nodes, for exactly the reason they do in `graph_y_max`: a
        route is free to climb above every node it passes over, and a panel scaled to its nodes
        alone would let those routes escape out of the top of it."""

        y_max = max((data['nodes'][nid]['position'][1] for nid in window_ids), default=0)

        for edge in self.active_edges(data):
            if edge['source'] not in window_ids or edge['target'] not in window_ids:
                continue
            for _x, y in (edge.get('route') or []):
                y_max = max(y_max, y)

        return y_max


    @staticmethod
    def edge_chain(pos_u, pos_v, route):
        """The graph-space (x, y) points an edge passes through: its two endpoints, with the
        edge's own route waypoints threaded between them. A route is how the graph says an
        edge detours around whatever sits between its endpoints instead of heading straight
        for them, so dropping it collapses a stepped path into one sloping line."""

        return [tuple(pos_u)] + [tuple(point) for point in (route or [])] + [tuple(pos_v)]


    def radial_projector(self, x_max, radius_of_y):
        """A (project, arc_px_between) pair for `project_chain` that lays a chain out around
        the main panel's circle, with `radius_of_y` saying which radius each y level sits at.
        Serves the main graph's growth zone and every genome's own track ring alike, since the
        two differ only in that mapping."""

        def project(x, y):
            return self.project_at_radius(x, radius_of_y(y), x_max)

        def arc_px_between(x0, x1, y):
            return abs(x1 - x0) / x_max * self.geom['angular_span'] * radius_of_y(y) if x_max else 0.0

        return project, arc_px_between


    def project_chain(self, chain, project, arc_px_between):
        """Screen points for a chain of graph-space (x, y) points, following the same rule
        anvi'o's interface draws every edge and every genome track with (`generate_svg()` in
        `anvio/data/interactive/js/pangenome-graph.js`): a segment between two points at the
        SAME y is an ARC along whatever curve that y level follows, while a segment that
        changes y cuts STRAIGHT across in screen space.

        Only the arcs get sampled, and that asymmetry is the whole point. Sampling a
        y-changing segment as well would bend it into a spiral, when what it should be is a
        straight radial step; not sampling a same-y segment leaves a chord cutting inside the
        circle it is supposed to trace, which is glaring once the two ends are far enough apart.

        `project` maps a graph-space point onto the canvas, and `arc_px_between` reports how
        many pixels of arc a same-y segment covers, which is what decides how finely it has to
        be sampled."""

        points = [project(*chain[0])]

        for (x0, y0), (x1, y1) in zip(chain, chain[1:]):
            if y0 != y1:
                points.append(project(x1, y1))
                continue

            steps = max(1, min(256, int(arc_px_between(x0, x1, y0) / ARC_SAMPLE_PX)))
            points.extend(project(x0 + (x1 - x0) * t / steps, y0) for t in range(1, steps + 1))

        return points


    def add_genome_track_shapes(self, shapes, frame_idx, x_max):
        """One ring per genome, each a compressed replica of that genome's OWN path through
        the pangenome graph, rather than a flat presence/absence indicator. This mirrors what
        anvi'o's interactive interface draws (`generate_svg()` in
        `anvio/data/interactive/js/pangenome-graph.js`): for each genome, its own ordered
        gene list (`node['synteny'][genome]`) is walked, consecutive genes are connected
        wherever a real graph edge exists between them, and the connecting line is drawn at
        a radius scaled into that genome's fixed-height ring using the SAME y (track/spire)
        value the main graph uses, along the same arcs and straight steps (see
        `project_chain`). A spire a genome participates in therefore shows up as a bump in ITS
        ring too, in the same angular place, just compressed. Ring height is
        fixed against the FINAL genome count (see `compute_geometry`), so an existing ring
        never resizes when a new one joins, it merely gains a new neighbor further out."""

        frame = self.frames[frame_idx]
        nodes = frame['data']['nodes']
        genomes = frame['genomes']
        if not genomes:
            return

        g = self.geom
        cx, cy = g['main_center']
        hole_r = g['hole_radius_px']
        track_height = g['fixed_track_height_px']
        y_max_frame = self.graph_y_max(frame['data'])

        # the route of every edge, keyed the way a genome walks it (source to target, the one
        # direction the interface looks it up in), so a missing key means the genome's walk has
        # no edge to follow here and its line has to break
        edge_routes = {(edge['source'], edge['target']): edge.get('route') or []
                       for edge in self.active_edges(frame['data'])}

        genome_track_state = frame['data']['states'].get('genome_tracks', {})
        bg_color_hex = self.genome_track_line_background or genome_track_state.get('background_color', '#F5F5F5')
        bg_color = self.hex_to_rgb(bg_color_hex)
        per_genome_state = genome_track_state.get('genomes', {})
        line_width = g['genome_track_line_width']

        theta_0, theta_1 = self.theta(0, x_max), self.theta(x_max, x_max)

        for i, genome_name in enumerate(genomes):
            r_inner = hole_r + i * track_height
            r_outer = r_inner + track_height * 0.82  # a thin gap between rings
            shapes.append(('polygon', self.wedge_polygon_points(cx, cy, r_inner, r_outer, theta_0, theta_1, n=96),
                          bg_color, 1.0))

            line_margin = track_height * 0.12
            r_lo, r_hi = r_inner + line_margin, r_outer - line_margin
            y_scale = (r_hi - r_lo) / y_max_frame if y_max_frame > 0 else 0.0

            synteny_map = {}
            for node_id, d in nodes.items():
                pos = d.get('synteny', {}).get(genome_name)
                if pos is not None:
                    synteny_map[int(pos)] = node_id

            track_color_hex = self.genome_track_line_color or per_genome_state.get(genome_name, {}).get('color') or '#000000'
            track_color = self.hex_to_rgb(track_color_hex)

            projector = self.radial_projector(x_max, lambda y: r_lo + y * y_scale)

            def flush(chain):
                if len(chain) >= 2:
                    shapes.append(('polyline', self.project_chain(chain, *projector),
                                  track_color, line_width, self.genome_track_line_opacity))

            # the genome's walk, gene by gene: gather graph-space points for as long as a real
            # edge carries it from one gene to the next, and break the line where none does
            chain = []
            for pos in sorted(synteny_map.keys()):
                node_i, node_j = synteny_map[pos], synteny_map.get(pos + 1)
                route = edge_routes.get((node_i, node_j)) if node_j is not None else None

                if route is None:
                    flush(chain)
                    chain = []
                    continue

                if not chain:
                    chain.append(tuple(nodes[node_i]['position']))
                chain.extend(tuple(point) for point in route)
                chain.append(tuple(nodes[node_j]['position']))

            flush(chain)


    def inset_window(self, frame_idx):
        """`(x_lo, x_hi, window_ids)` for a frame: the two flank positions that bracket the
        magnified locus, and every node between them.

        This has to be decided from THIS frame's own flank positions rather than the final
        frame's, since positions come from a fresh layout every time genomes change (see
        `compute_geometry`'s docstring on why), so a backbone position number from one frame
        means nothing in another. Only the zoom scale (`compute_inset_geometry`) is held fixed
        across frames."""

        left, right = self.inset_flank_ids
        nodes = self.frames[frame_idx]['data']['nodes']
        x_lo, x_hi = sorted([nodes[left]['position'][0], nodes[right]['position'][0]])

        return x_lo, x_hi, {nid for nid, d in nodes.items() if x_lo <= d['position'][0] <= x_hi}


    def add_inset_shapes(self, shapes, frame_idx):
        """The magnified panel: its chrome, then the subgraph between the two flanking nodes,
        drawn with the very same route-following edges the main panel uses (see `edge_chain`
        and `project_chain`) so the panel reads as that piece of the graph enlarged rather
        than as a differently-wired picture of it."""

        ox, oy, panel_w, panel_h = self.geom['panel_box']

        shapes.append(('round_rect', ox, oy, panel_w, panel_h, self.px(INSET_CORNER_RADIUS),
                      self.tint_color(), 1.0))
        shapes.append(('round_rect_outline', ox, oy, panel_w, panel_h, self.px(INSET_CORNER_RADIUS),
                      BLUE, self.border_width_px()))

        nodes = self.frames[frame_idx]['data']['nodes']
        x_lo, x_hi, window_ids = self.inset_window(frame_idx)
        inset = self.geom['inset']

        for edge in self.active_edges(self.frames[frame_idx]['data']):
            u, v = edge['source'], edge['target']
            if u not in window_ids or v not in window_ids:
                continue
            # the panel lays its levels out on straight lines, so unlike the main panel there
            # is no arc to sample along: every point of the chain is a corner of the drawn path
            chain = self.edge_chain(nodes[u]['position'], nodes[v]['position'], edge.get('route'))
            pixels = [self.project_inset(x, y, x_lo, x_hi) for x, y in chain]
            if len(pixels) >= 2:
                shapes.append(('polyline', pixels, inset['edge_color'], inset['edge_w'], inset['edge_opacity']))

        node_color = self.hex_to_rgb(self.inset_graph_node_color) if self.inset_graph_node_color else None

        for node_id in window_ids:
            d = nodes[node_id]
            x, y = self.project_inset(*d['position'], x_lo, x_hi)
            color = node_color or self.hex_to_rgb(self.type_colors.get(d.get('type'), '#999999'))
            shapes.append(('circle', x, y, inset['node_r'], color,
                          inset['node_edge_color'], inset['node_edge_width']))


    def add_inset_marker_shapes(self, shapes, frame_idx, x_max, n_genomes):
        """The counterpart of the inset panel on the main graph: a wedge over exactly the
        stretch of graph the panel magnifies, in the panel's own fill and outline, so the eye
        can pair the two.

        It reaches from just inside the backbone ring out past the tallest thing in the window,
        and is added before the graph itself so it sits behind the edges and nodes rather than
        over them."""

        x_lo, x_hi, window_ids = self.inset_window(frame_idx)
        if not window_ids:
            return

        backbone_radius = self.backbone_radius(n_genomes)
        disty = self.geom['disty_px']
        y_hi = self.inset_window_y_max(self.frames[frame_idx]['data'], window_ids)

        # the flanking nodes sit ON the boundary, so the wedge is padded out past them by a
        # node's own footprint, or it would slice them in half
        pad = max(self.geom['main_graph_node_size'] * 2.0, self.px(6))
        r_inner = max(0.0, backbone_radius - pad)
        r_outer = backbone_radius + y_hi * disty + pad

        # `theta` runs backwards (see its docstring), so x_hi gives the smaller angle
        x_pad = pad / disty if disty > 0 else 0.0
        theta_hi = self.theta(max(0.0, x_lo - x_pad), x_max)
        theta_lo = self.theta(x_hi + x_pad, x_max)

        cx, cy = self.geom['main_center']
        points = self.wedge_polygon_points(cx, cy, r_inner, r_outer, theta_lo, theta_hi, n=96)

        shapes.append(('polygon', points, self.tint_color(), 1.0))
        shapes.append(('polygon_outline', points, BLUE, self.border_width_px()))


    def tint_color(self):
        return tuple(round((1 - TINT_ALPHA) * 255 + TINT_ALPHA * c) for c in BLUE)


    def border_width_px(self):
        return max(2.0, self.px(6))


    def counter_block_geometry(self):
        """The type sizes of the genome counter block: a big bold genome count, the rest of the
        sentence trailing it in a smaller muted face, and a progress bar underneath. Where the
        block SITS is solved separately, by `compute_counter_block_layout`.

        `max_text_size` is a ceiling rather than the size actually used: the trailing text is
        sized to fit the column by `compute_counter_text_geometry`, and can end up well below
        this ceiling when genome names are long."""

        return {'number_size': self.px(230), 'max_text_size': self.px(96), 'gap': self.px(26),
                'x_nudge': self.px(8), 'bar_h': self.px(9), 'description_size': self.px(54),
                'description_line_height': self.px(50)}


    def compute_counter_text_geometry(self):
        """How much room the text trailing the big genome count gets, and what size it is set
        at. Both are pinned once here and reused by every frame.

        That text has to stay on ONE line no wider than the progress bar beneath it, and how
        much room it needs is entirely a property of the pan-graph-db at hand: a dataset of
        'HIMB122's needs a fraction of what a dataset of 'Pseudoalteromonas_sp_BSi20429's
        does. So the size is solved for against the LONGEST line the video will ever show, and
        then held fixed, since text that changed size as genomes came and went would read as
        jitter.

        Names can be long enough that no readable size fits them, which is what the floor
        below is for: past that point the size stops shrinking and `genome_counter_text` clips
        the name instead."""

        c = self.counter_block_geometry()

        # the room is measured against the WIDEST count the video will show (the final one,
        # which has the most digits) rather than the count in whichever frame is being drawn,
        # so that a clipped name loses exactly the same characters in every single frame
        number_width = self.get_font(c['number_size'], 'bold').getlength(str(self.total_genomes_rendered))
        room = self.geom['col_w'] - number_width - c['gap']

        longest_name = max((frame['genomes'][-1] for frame in self.frames), key=len)
        longest_line = self.genome_counter_head(self.total_genomes_rendered) + f" (added '{longest_name}')"

        size, floor, step = c['max_text_size'], self.px(48), self.px(2)
        while size > floor and self.get_font(size, 'regular').getlength(longest_line) > room:
            size -= step

        self.geom['counter_text_size'] = max(size, floor)
        self.geom['counter_text_room'] = room


    @staticmethod
    def wrap_text(text, font, width):
        """`text` broken into as few lines as fit within `width` pixels, one word at a time."""

        lines, current = [], ''

        for word in text.split():
            candidate = f'{current} {word}'.strip()
            if not current or font.getlength(candidate) <= width:
                current = candidate
            else:
                lines.append(current)
                current = word

        if current:
            lines.append(current)

        return lines


    def compute_counter_block_layout(self):
        """Where each piece of the genome counter block sits vertically: the counter line, the
        progress bar under it, and, in the 'tower' arrangement, the inset description under
        that.

        The three stack UPWARD from the top of the inset panel with one single gap
        (COUNTER_GAP) repeated between them, so the counter line stands as far above the bar as
        the bar stands above the description, and the description stands that same distance
        above the panel itself. The gaps are measured between real INK extents rather than
        between type sizes or baselines, since what the eye compares is where the marks
        actually stop: a line of text with no descender in it hangs differently from one with
        three.

        Solving this once, here, rather than per frame is also what keeps the block from
        drifting: the count gains a digit and the genome names change length as the video
        runs, and the block would creep with them if each frame measured its own strings."""

        c = self.counter_block_geometry()
        number_font = self.get_font(c['number_size'], 'bold')
        text_font = self.get_font(self.geom['counter_text_size'], 'regular')

        # PIL anchors text at the ascender, so putting the trailing text on the count's own
        # baseline is a matter of matching the two ascents (and then lifting it a touch)
        text_offset = (number_font.getmetrics()[0] - text_font.getmetrics()[0]
                       - c['number_size'] * COUNTER_TEXT_LIFT)

        layout = {'text_offset': text_offset}

        # the final frame's line is the one to measure against: it carries the widest count,
        # and since names are clipped to a fixed width (see `compute_counter_text_geometry`)
        # no other frame's line can reach lower
        final_label = str(self.total_genomes_rendered)
        final_counter = self.genome_counter_text(self.total_genomes_rendered, self.frames[-1]['genomes'][-1],
                                                text_font, self.geom['counter_text_room'])
        counter_ink_bottom = max(number_font.getbbox(final_label)[3],
                                 text_offset + text_font.getbbox(final_counter)[3])

        gap = self.px(COUNTER_GAP)
        panel_top = self.geom['panel_box'][1]
        desc_font = self.get_font(c['description_size'], 'regular')

        lines = self.wrap_text(self.inset_description, desc_font, self.geom['col_w']) if self.inset_description else []

        # The description's slot is reserved whether or not there is anything to put in it, so
        # that dropping the inset moves nothing above it: the description and the panel simply
        # stop being drawn and everything else stays put. An empty slot is one line tall,
        # measured off a string carrying both an ascender and a descender so that it takes up
        # what a real line of it would.
        ruler = lines or ['Ag']
        ink_top = desc_font.getbbox(ruler[0])[1]
        ink_bottom = (len(ruler) - 1) * c['description_line_height'] + desc_font.getbbox(ruler[-1])[3]

        bar_y = panel_top - 2 * gap - c['bar_h'] - ink_bottom + ink_top

        layout['description_lines'] = lines
        layout['description_y'] = bar_y + c['bar_h'] + gap - ink_top
        layout['bar_y'] = bar_y
        layout['number_y'] = bar_y - gap - counter_ink_bottom

        self.geom['counter'] = layout


    @staticmethod
    def genome_counter_head(ngen):
        """The part of the counter line that only depends on the total genome count."""

        return 'of ' + str(ngen) + ' genome' + ('' if ngen == 1 else 's')


    def genome_counter_text(self, ngen, genome_added, text_font, room):
        """Everything that trails the big genome count: 'of 42 genomes', then the name of the
        genome that just joined the graph. A `genome_added` of None leaves the name out, which
        is what the finished graph holds on at the end of the video, once there is no longer a
        genome that just arrived.

        `compute_counter_text_geometry` has already picked a size that fits the longest name
        in this pan-graph-db wherever it could. Where it could not, the name is the only part
        of the line that can be any length, so the name is what gives: it gets clipped, with
        an ellipsis standing in for whatever had to go. A name can be long enough that not
        even one character of it fits, in which case it is dropped and only the count is
        shown."""

        head = self.genome_counter_head(ngen)

        if genome_added is None:
            return head

        def with_name(name):
            return f"{head} (added '{name}')"

        if text_font.getlength(with_name(genome_added)) <= room:
            return with_name(genome_added)

        for num_characters in range(len(genome_added) - 1, 0, -1):
            if text_font.getlength(with_name(genome_added[:num_characters] + '...')) <= room:
                return with_name(genome_added[:num_characters] + '...')

        return head


    def add_text_ops(self, texts, shapes, label, ngen, genome_added):
        """The genome counter block (see `counter_block_geometry`), plus an optional three-part
        title above it and the inset description below it. Every one of these sits in the same
        place whether or not there is an inset; without one, the description is the only piece
        that goes missing."""

        px, x, col_w = self.px, self.geom['col_x'], self.geom['col_w']

        def text(tx, ty, size, role, color, s):
            texts.append(('text', tx, ty, size, role, color, s))

        def rect(rx, ry, rw, rh, color):
            shapes.append(('rect', rx, ry, rw, rh, color, 1.0))

        if self.title_kicker:
            text(x, px(132), px(52), 'medium', MUTED, self.title_kicker)
        if self.title:
            text(x - px(6), px(196), px(112), 'italic', INK, self.title)
        if self.subtitle:
            text(x, px(344), px(54), 'regular', MUTED, self.subtitle)

        c, layout = self.counter_block_geometry(), self.geom['counter']

        number_x = x - c['x_nudge']
        number_width = self.get_font(c['number_size'], 'bold').getlength(str(label))
        text(number_x, layout['number_y'], c['number_size'], 'bold', INK, str(label))

        text_size = self.geom['counter_text_size']
        counter = self.genome_counter_text(ngen, genome_added, self.get_font(text_size, 'regular'),
                                          self.geom['counter_text_room'])
        text(number_x + number_width + c['gap'], layout['number_y'] + layout['text_offset'],
             text_size, 'regular', MUTED, counter)

        by, bh = layout['bar_y'], c['bar_h']
        rect(x, by, col_w, bh, FAINT)
        rect(x, by, col_w * label / ngen if ngen else 0, bh, INK)

        # --inset-description lives in the slot between the progress bar and the inset panel,
        # in the same style as --subtitle. That slot is reserved either way, so this drawing
        # nothing is what the no-inset case looks like.
        for i, line in enumerate(layout['description_lines']):
            text(x, layout['description_y'] + i * c['description_line_height'],
                 c['description_size'], 'regular', MUTED, line)


    @staticmethod
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


    # ------------------------------------------------------------------
    # The two renderers. Both walk the exact same scene.
    # ------------------------------------------------------------------

    def run_in_parallel(self, function, items):
        """`function` called once per item on --num-threads workers, returning only once every
        one of them is done.

        THREADS, for work that is plainly CPU-bound, which looks like the wrong call until you
        look at where a frame's time actually goes. The great majority of it is inside Pillow's C
        code — the supersample downsample and the PNG encode — and both of those let go of the
        GIL while they run. What holds it is the Python loop walking the scene, and that is the
        minority of the work. So the speedup is real but sublinear (nearer 3x on eight threads
        than 8x), and it costs none of the pickling, memory duplication or inter-process plumbing
        that `multiprocess` would ask for in return for roughly twice as much again.

        Nothing here touches `self.progress`, which is not thread-safe. Callers report progress
        themselves, from the main thread, at whatever granularity suits them.

        Whatever a worker raises is re-raised here rather than being left to quietly become a
        half-written frame sequence that ffmpeg would then happily encode."""

        if self.num_threads == 1:
            for item in items:
                function(item)
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for future in [executor.submit(function, item) for item in items]:
                future.result()


    def rasterize_scenes(self, scenes):
        """Every scene rasterized, in the order they were handed over.

        Unlike the frame workers, which write their image and forget it, these images are all
        KEPT: the timeline reaches back for a genome's settled image long after its own turn is
        over, both to dissolve out of it and to hold on it."""

        images = []

        if self.num_threads == 1:
            for scene in scenes:
                images.append(self.rasterize_scene(scene))
                self.progress.increment()

            return images

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self.rasterize_scene, scene) for scene in scenes]

            # collected in submission order, and the progress bar is touched only from here,
            # which is to say only from the main thread
            for future in futures:
                images.append(future.result())
                self.progress.increment()

        return images


    def count_hard_linked_frames(self, effect_frames):
        """How many frames of the timeline are a repeat of the frame before them, and are
        therefore hard-linked into place rather than encoded all over again (see `write_image`
        in `build_video`)."""

        _dissolve_frames, arrival_frames, turn_frames = self.genome_turn_frames(effect_frames)

        linked = max(1, int(self.hold_first_seconds * self.fps)) - 1
        linked += (len(self.frames) - 1) * (max(1, turn_frames - arrival_frames) - 1)
        linked += max(1, int(self.hold_last_seconds * self.fps)) - 1

        return linked


    @staticmethod
    def draw_polyline(img, draw, pts, color, width, opacity):
        """One polyline, composited exactly ONCE however many points it has.

        Handing a translucent ink straight to `ImageDraw.line` does not do that. Pillow lays a
        multi-point line down segment by segment, compositing each one separately, so wherever
        consecutive segments overlap — which is everywhere, since a thick line's segments have to
        overlap to join cleanly — the ink is blended over itself again and again. On a two-point
        line that is invisible. On the lines this program actually draws it is fatal: `project_chain`
        samples an arc every ARC_SAMPLE_PX pixels, so a long sweep arrives here as hundreds of
        points and ANY opacity saturates to solid. --main-graph-edge-opacity and
        --genome-track-line-opacity were both quietly doing nothing as a result.

        So the line is stroked into a plain mask instead, where drawing sets pixels rather than
        blending them and self-overlap therefore costs nothing, and the mask is composited in one
        go. Overlap BETWEEN two separate polylines still darkens, which is both what one expects
        of two translucent strokes crossing and what `scene_to_svg` does with the same scene.

        The mask covers only the line's own bounding box, so what this costs tracks the ink drawn
        rather than the size of the canvas."""

        from PIL import Image, ImageDraw

        if opacity >= 1.0:
            draw.line(pts, fill=color, width=width, joint='curve')
            return

        alpha = int(255 * max(0.0, min(1.0, opacity)))
        if alpha == 0:
            return

        pad = width // 2 + 2
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        x0, y0 = max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad)
        x1, y1 = min(img.width, int(max(xs)) + pad + 1), min(img.height, int(max(ys)) + pad + 1)

        if x1 <= x0 or y1 <= y0:
            return

        mask = Image.new('L', (x1 - x0, y1 - y0), 0)
        ImageDraw.Draw(mask).line([(x - x0, y - y0) for x, y in pts], fill=alpha, width=width, joint='curve')
        img.paste(color, (x0, y0), mask)


    def rasterize_scene(self, scene):
        from PIL import Image, ImageDraw

        ss = self.supersample

        # The canvas is RGB while the drawing mode is RGBA, and that pairing is the ONLY one
        # Pillow blends translucent shapes through: handed an RGBA canvas it overwrites the
        # pixel outright, alpha and all, so every opacity in the scene would be thrown away
        # and only the SVG would honour them.
        img = Image.new('RGB', (int(self.canvas_w * ss), int(self.canvas_h * ss)), self.background_color)
        draw = ImageDraw.Draw(img, 'RGBA')

        for shape in scene['shapes']:
            kind = shape[0]
            if kind == 'polygon':
                _, pts, color, opacity = shape
                draw.polygon([(x * ss, y * ss) for x, y in pts], fill=color + (int(255 * opacity),))
            elif kind == 'polyline':
                _, pts, color, width, opacity = shape
                self.draw_polyline(img, draw, [(x * ss, y * ss) for x, y in pts], color,
                                   max(1, int(round(width * ss))), opacity)
            elif kind == 'circle':
                _, x, y, r, fill, outline, outline_w = shape
                x, y, r, outline_w = x * ss, y * ss, r * ss, outline_w * ss
                draw.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline=outline, width=max(1, int(round(outline_w))))
            elif kind == 'rect':
                _, x, y, w, h, fill, opacity = shape
                draw.rectangle([x * ss, y * ss, (x + w) * ss, (y + h) * ss], fill=fill + (int(255 * opacity),))
            elif kind == 'rect_outline':
                _, x, y, w, h, color, width = shape
                draw.rectangle([x * ss, y * ss, (x + w) * ss, (y + h) * ss], outline=color, width=max(1, int(round(width * ss))))
            elif kind == 'round_rect':
                _, x, y, w, h, radius, fill, opacity = shape
                draw.rounded_rectangle([x * ss, y * ss, (x + w) * ss, (y + h) * ss], radius=radius * ss,
                                      fill=fill + (int(255 * opacity),))
            elif kind == 'round_rect_outline':
                _, x, y, w, h, radius, color, width = shape
                draw.rounded_rectangle([x * ss, y * ss, (x + w) * ss, (y + h) * ss], radius=radius * ss,
                                      outline=color, width=max(1, int(round(width * ss))))
            elif kind == 'polygon_outline':
                _, pts, color, width = shape
                closed = [(x * ss, y * ss) for x, y in pts] + [(pts[0][0] * ss, pts[0][1] * ss)]
                draw.line(closed, fill=color, width=max(1, int(round(width * ss))), joint='curve')
            elif kind == 'fading_circle':
                _, x, y, r, fill, outline, outline_w, opacity = shape
                alpha = int(255 * max(0.0, min(1.0, opacity)))
                x, y, r, outline_w = x * ss, y * ss, r * ss, outline_w * ss
                draw.ellipse([x - r, y - r, x + r, y + r], fill=fill + (alpha,),
                            outline=outline + (alpha,), width=max(1, int(round(outline_w))))

        for text_op in scene['texts']:
            _, x, y, size, role, color, s = text_op
            font = self.get_font(size * ss, role)
            draw.text((x * ss, y * ss), s, fill=color, font=font)

        if ss > 1:
            img = img.resize((self.canvas_w, self.canvas_h), Image.LANCZOS)

        return img


    def svg_font_family(self):
        """The font-family an exported SVG names.

        One family, not a fallback stack. An editor shows whatever is written here verbatim in
        its own font box, so a stack has to be deleted and retyped by hand before it will do
        anything, every single time the file is opened. Naming a single family is only honest if
        that family is actually on the machine, which is what the presence of its file settles,
        so the stack is kept in reserve for when it is not."""

        path, _faces, family = FONT_SETS[self.font_choice]

        return family if os.path.exists(path) else FALLBACK_FONT_FAMILY


    def scene_to_svg(self, scene):
        family = self.svg_font_family()

        def rgb(c):
            return f'rgb({c[0]},{c[1]},{c[2]})'

        def escape(s):
            return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.canvas_w}" height="{self.canvas_h}" '
                f'viewBox="0 0 {self.canvas_w} {self.canvas_h}">',
                f'<rect x="0" y="0" width="{self.canvas_w}" height="{self.canvas_h}" fill="{self.background_color}"/>']

        for shape in scene['shapes']:
            kind = shape[0]
            if kind == 'polygon':
                _, pts, color, opacity = shape
                pts_str = ' '.join(f'{px:.2f},{py:.2f}' for px, py in pts)
                parts.append(f'<polygon points="{pts_str}" fill="{rgb(color)}" fill-opacity="{opacity:.3f}"/>')
            elif kind == 'polyline':
                _, pts, color, width, opacity = shape
                pts_str = ' '.join(f'{px:.2f},{py:.2f}' for px, py in pts)
                parts.append(f'<polyline points="{pts_str}" fill="none" stroke="{rgb(color)}" '
                            f'stroke-width="{width:.2f}" stroke-opacity="{opacity:.3f}" '
                            f'stroke-linecap="round" stroke-linejoin="round"/>')
            elif kind == 'circle':
                _, x, y, r, fill, outline, outline_w = shape
                parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{rgb(fill)}" '
                            f'stroke="{rgb(outline)}" stroke-width="{outline_w:.2f}"/>')
            elif kind == 'rect':
                _, x, y, w, h, fill, opacity = shape
                parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                            f'fill="{rgb(fill)}" fill-opacity="{opacity:.3f}"/>')
            elif kind == 'rect_outline':
                _, x, y, w, h, color, width = shape
                parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                            f'fill="none" stroke="{rgb(color)}" stroke-width="{width:.2f}"/>')
            elif kind == 'round_rect':
                _, x, y, w, h, radius, fill, opacity = shape
                parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                            f'rx="{radius:.2f}" ry="{radius:.2f}" fill="{rgb(fill)}" '
                            f'fill-opacity="{opacity:.3f}"/>')
            elif kind == 'round_rect_outline':
                _, x, y, w, h, radius, color, width = shape
                parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                            f'rx="{radius:.2f}" ry="{radius:.2f}" fill="none" '
                            f'stroke="{rgb(color)}" stroke-width="{width:.2f}"/>')
            elif kind == 'polygon_outline':
                _, pts, color, width = shape
                pts_str = ' '.join(f'{px:.2f},{py:.2f}' for px, py in pts)
                parts.append(f'<polygon points="{pts_str}" fill="none" stroke="{rgb(color)}" '
                            f'stroke-width="{width:.2f}" stroke-linejoin="round"/>')
            elif kind == 'fading_circle':
                _, x, y, r, fill, outline, outline_w, opacity = shape
                parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{rgb(fill)}" '
                            f'fill-opacity="{opacity:.3f}" stroke="{rgb(outline)}" '
                            f'stroke-width="{outline_w:.2f}" stroke-opacity="{opacity:.3f}"/>')

        for text_op in scene['texts']:
            _, x, y, size, role, color, s = text_op
            weight = 700 if role in BOLD_ROLES else (600 if role == 'demi' else (500 if role == 'medium' else 400))
            style = 'italic' if role in ITALIC_ROLES else 'normal'
            # PIL anchors text at the glyph ascender (top-left, anchor 'la'); SVG anchors at
            # the baseline, so nudge down by the font's actual ascent, which PIL can report
            # exactly since it has the real font loaded.
            ascent, _descent = self.get_font(size, role).getmetrics()
            baseline_y = y + ascent
            parts.append(f'<text x="{x:.2f}" y="{baseline_y:.2f}" font-family="{family}" font-size="{size:.2f}" '
                        f'font-weight="{weight}" font-style="{style}" fill="{rgb(color)}">{escape(s)}</text>')

        parts.append('</svg>')
        return '\n'.join(parts)


    def export_still(self, scene, key):
        svg_path, png_path = self.still_paths[key]
        with open(svg_path, 'w') as f:
            f.write(self.scene_to_svg(scene))
        self.rasterize_scene(scene).save(png_path)
        self.run.info(f"Still frame ({key})", f"{svg_path}, {png_path}")


    def user_set_cosmetic(self, dest):
        """Whether a COSMETICS flag carries a value other than the one it falls back to when
        left alone. Reported back in a different color, so a glance at the section separates
        what was asked for from what was worked out from the data."""

        if dest not in self.cosmetic_defaults:
            return False

        return self.args.__dict__.get(dest) != self.cosmetic_defaults[dest]


    def report_cosmetics(self):
        """No formula for node/edge ink can look right for every combination of genome count
        and locus shape (see the docstrings in `compute_geometry` for exactly where each
        default comes from), so every one of these is also a plain CLI flag. This prints back
        each resolved value (default or user-provided) right before rendering starts, so the
        flag to pass next time is always one line away."""

        self.run.warning(None, header="COSMETICS", lc="green")
        g = self.geom

        # anvi'o's own dot-padding aligns the ':' at column `self.run.width` (default 45),
        # which every one of these labels (a description plus the literal flag name) blows
        # past, collapsing the dots entirely. Widen it just for this section.
        original_width = self.run.width
        self.run.width = 78

        def show(label, flag, value):
            text = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
            self.run.info(f"{label} (`--{flag}`)", text,
                          mc='green' if self.user_set_cosmetic(flag.replace('-', '_')) else 'yellow')

        show("Main graph node size", "main-graph-node-size", g['main_graph_node_size'])
        show("Main graph node color", "main-graph-node-color", self.main_graph_node_color or AUTO_FROM_DB)
        show("Main graph new node effect", "main-graph-new-node-effect", self.main_graph_new_node_effect)
        show("Main graph nodes fly in", "main-graph-nodes-fly-in", 'yes' if self.main_graph_nodes_fly_in else 'no')
        show("Main graph node edge width", "main-graph-node-edge-width", g['main_graph_node_edge_width'])
        show("Main graph node edge color", "main-graph-node-edge-color", self.main_graph_node_edge_color)
        show("Main graph edge width", "main-graph-edge-width", g['main_graph_edge_width'])
        show("Main graph edge color", "main-graph-edge-color", self.main_graph_edge_color)
        show("Main graph edge opacity", "main-graph-edge-opacity", g['main_graph_edge_opacity'])

        if g.get('has_inset'):
            inset = g['inset']
            show("Inset graph level height", "inset-graph-level-height", inset['level_height'])
            show("Inset graph node size", "inset-graph-node-size", inset['node_r'])
            show("Inset graph node color", "inset-graph-node-color", self.inset_graph_node_color or AUTO_FROM_DB)
            show("Inset graph new node effect", "inset-graph-new-node-effect", self.inset_graph_new_node_effect)
            show("Inset graph node edge width", "inset-graph-node-edge-width", inset['node_edge_width'])
            show("Inset graph node edge color", "inset-graph-node-edge-color", self.inset_graph_node_edge_color)
            show("Inset graph edge width", "inset-graph-edge-width", inset['edge_w'])
            show("Inset graph edge color", "inset-graph-edge-color", self.inset_graph_edge_color)
            show("Inset graph edge opacity", "inset-graph-edge-opacity", inset['edge_opacity'])

        if self.show_genome_tracks:
            show("Genome track line width", "genome-track-line-width", g['genome_track_line_width'])
            show("Genome track line color", "genome-track-line-color", self.genome_track_line_color or "auto (per genome, from pan-graph-db)")
            show("Genome track line opacity", "genome-track-line-opacity", self.genome_track_line_opacity)
            show("Genome track line background", "genome-track-line-background", self.genome_track_line_background or AUTO_FROM_DB)

        self.run.width = original_width


    def build_video(self):
        from PIL import Image

        self.compute_frames()
        self.compute_geometry()
        self.report_cosmetics()

        self.run.warning(None, header="BUILDING SCENES", lc="green")
        self.progress.new("Building per-genome scenes", progress_total_items=len(self.frames))

        scenes = []
        for i in range(len(self.frames)):
            self.progress.update(f"{self.frames[i]['n']} genome(s)")
            scenes.append(self.build_scene(i))
            self.progress.increment()

        self.progress.end()

        # the video ends by holding the finished graph, and by then no genome has just
        # arrived, so the last thing on screen is the final frame with that name taken back
        # off the counter line
        final_scene = self.build_scene(len(self.frames) - 1, name_the_genome_added=False)

        if self.export_stills:
            self.export_still(scenes[0], 'first-frame')
            self.export_still(final_scene, 'last-frame')

        if self.dry_run:
            self.run.info_single("--dry-run was passed, so stopping here, right before assembling the video. "
                                 "The first/last-frame stills above already reflect the current cosmetics.",
                                 nl_before=1, mc="green")
            return False

        effect_frames = self.overlay_frames()
        total_frames = self.count_video_frames(effect_frames)

        # one rasterization per genome, plus the nameless final still, plus every frame of
        # every new-node effect. The rest of the timeline reuses those images, so this is what
        # the render time actually tracks
        to_rasterize = len(scenes) + 1
        if effect_frames:
            to_rasterize += self.genome_turn_frames(effect_frames)[1] * (len(scenes) - 1)

        self.run.warning(None, header="ASSEMBLING VIDEO", lc="green")
        self.run.info("New node effect, main graph", self.main_graph_new_node_effect,
                      mc='green' if self.user_set_cosmetic('main_graph_new_node_effect') else 'yellow')
        if self.geom.get('has_inset'):
            self.run.info("New node effect, inset graph", self.inset_graph_new_node_effect,
                          mc='green' if self.user_set_cosmetic('inset_graph_new_node_effect') else 'yellow')
        if self.main_graph_nodes_fly_in:
            self.run.info("Nodes that fly in", f"{pp(sum(len(a) for a in self.new_nodes_per_frame))} across the "
                          f"whole video, every node that is ever new to the graph")
        if effect_frames:
            longest = max(self.overlay_duration_fractions())
            self.run.info("Effect frames per genome", f"{effect_frames} "
                          f"({longest:g} of --seconds-per-genome, at {self.fps} fps), starting on the "
                          f"very first frame the genome fades in on")
        self.run.info("Frames to rasterize", pp(to_rasterize) + (" (one per genome, plus every effect frame)"
                                                                if effect_frames else " (one per genome)"))
        self.run.info("Threads to rasterize with", f"{self.num_threads} (`--num-threads`)")

        hard_linked = self.count_hard_linked_frames(effect_frames)
        self.run.info("Frames in the video", f"{pp(total_frames)} ({pp(hard_linked)} of them repeats of the "
                      f"frame before them, hard-linked into place rather than encoded again)")
        self.run.info("Video duration", f"{total_frames / self.fps:.1f} seconds")

        self.progress.new("Rasterizing frames", progress_total_items=len(scenes) + 1)
        images = self.rasterize_scenes(scenes + [final_scene])
        self.progress.end()

        final_image = images.pop()

        frames_dir = self.keep_frames_dir or tempfile.mkdtemp(prefix="pangraph_video_")
        if self.keep_frames_dir:
            filesnpaths.gen_output_directory(frames_dir)

        self.progress.new("Writing frame sequence", progress_total_items=len(self.frames))

        frame_counter = [0]

        def frame_path(index):
            return os.path.join(frames_dir, f"frame_{index:06d}.png")

        def write_image(image, repeats=1):
            """The next `repeats` frames of the timeline, every one of them this same image.

            A repeat is the very same picture again, so it is encoded ONCE and the frames behind
            it are hard links to that one file rather than fresh PNGs of byte-identical content.
            Encoding a 4K PNG costs a good fraction of a second and a link costs nothing at all,
            and the timeline is mostly repeats wherever nothing is moving: --hold-last-seconds by
            itself accounts for 120 of them at the default frame rate, and a run with no effect
            asked for holds every genome for a dozen more. ffmpeg reads a link like any other
            file."""

            if repeats < 1:
                return

            first_path = frame_path(frame_counter[0])
            image.save(first_path)
            frame_counter[0] += 1

            for _ in range(repeats - 1):
                path = frame_path(frame_counter[0])
                try:
                    os.link(first_path, path)
                except OSError:
                    # not every filesystem does hard links, and a copy is still far cheaper than
                    # encoding the same image all over again
                    shutil.copyfile(first_path, path)
                frame_counter[0] += 1

        dissolve_frames, arrival_frames, turn_frames = self.genome_turn_frames(effect_frames)

        def dissolve(img_a, img_b):
            start = frame_counter[0]

            def blend_and_write(step):
                Image.blend(img_a, img_b, step / (dissolve_frames + 1)).save(frame_path(start + step - 1))

            self.run_in_parallel(blend_and_write, range(1, dissolve_frames + 1))
            frame_counter[0] = start + dissolve_frames

        def play_genome(i):
            """One genome's turn: it fades in with its effect ALREADY running over it, the
            effect finishes over the settled graph, and whatever is left of the turn is held
            still. Only the overlay is recomputed per frame, so nothing underneath is built
            twice.

            Every frame of the turn is independent of every other one, which is what lets them be
            rasterized side by side (see `run_in_parallel`). Each worker saves its own frame
            straight to the index that was worked out for it before any of them started, so
            nothing has to be handed back in order, and no more than --num-threads frames are
            ever in memory at once."""

            # the arriving genome's own nodes are in the air for the whole of this, so the base
            # underneath them is this frame's scene WITHOUT them. Everything else about it is
            # identical to `scenes[i]`, which is what the rest of the turn is held on.
            base = self.build_scene(i, arrivals_in_flight=True) if self.main_graph_nodes_fly_in else scenes[i]

            start = frame_counter[0]

            def render_and_write(step):
                overlay = self.arrival_overlay_shapes(i, min(1.0, step / effect_frames))
                image = self.rasterize_scene({'shapes': base['shapes'] + overlay,
                                             'texts': base['texts']})
                if step < dissolve_frames:
                    image = Image.blend(images[i - 1], image, (step + 1) / (dissolve_frames + 1))
                image.save(frame_path(start + step))

            self.run_in_parallel(render_and_write, range(arrival_frames))
            frame_counter[0] = start + arrival_frames

            write_image(images[i], repeats=max(1, turn_frames - arrival_frames))

        def note(i):
            of = f"genome {self.frames[i]['n']}/{self.total_genomes_rendered}"
            self.progress.update(f"{of}, rendering {effect_frames} effect frames" if effect_frames else of)
            self.progress.increment()

        note(0)
        write_image(images[0], repeats=max(1, int(self.hold_first_seconds * self.fps)))

        for i in range(1, len(images)):
            note(i)
            if effect_frames:
                play_genome(i)
            else:
                # nothing to lay over the graph, so the fade is a plain blend of two ready
                # images and the whole of the rest of the turn is one of them repeated
                dissolve(images[i - 1], images[i])
                write_image(images[i], repeats=max(1, turn_frames - dissolve_frames))

        dissolve(images[-1], final_image)
        write_image(final_image, repeats=max(1, int(self.hold_last_seconds * self.fps)))

        self.progress.end()

        self.encode_video(frames_dir)

        if not self.keep_frames_dir:
            shutil.rmtree(frames_dir, ignore_errors=True)
        else:
            self.run.info("Frame PNGs kept at", frames_dir)

        return True


    def encode_video(self, frames_dir):
        log_file_path = os.path.join(tempfile.gettempdir(), "anvi-script-gen-pan-graph-video-ffmpeg.log")

        cmdline = ["ffmpeg", "-y",
                  "-framerate", str(self.fps),
                  "-i", os.path.join(frames_dir, "frame_%06d.png"),
                  "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                  "-shortest",
                  "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
                  "-c:a", "aac", "-b:a", "128k",
                  "-movflags", "+faststart",
                  self.output_file]

        self.progress.new("Encoding with ffmpeg")
        self.progress.update("this can take a moment for long videos")
        utils.run_command(cmdline, log_file_path)
        self.progress.end()

        if not filesnpaths.is_file_exists(self.output_file, dont_raise=True):
            raise ConfigError(f"ffmpeg did not produce an output file. Check the log at '{log_file_path}' "
                              f"for what went wrong.")


    def process(self):
        completed = self.build_video()
        if completed:
            self.run.info_single(f"Done! Your video is ready: {self.output_file}", nl_before=1, mc="green")


def main():
    args = get_args()

    try:
        video_generator = PanGraphVideoGenerator(args)
        video_generator.process()
    except ConfigError as e:
        print(e)
        sys.exit(-1)
    except FilesNPathsError as e:
        print(e)
        sys.exit(-2)


def get_args():
    parser = ArgumentParser(description=__description__)

    groupA = parser.add_argument_group('INPUT', "Where the pangenome graph data comes from.")
    groupA.add_argument(*anvio.A('pan-graph-db'), **anvio.K('pan-graph-db'))
    groupA.add_argument('--component', default='CP_0001', help="Which weakly connected component to render "
                        "(as named in the pan-graph-db; 'CP_0001' is the largest by default).")
    groupA.add_argument('--genomes-of-interest', default=None, help="Either a comma-separated list of genome "
                        "names, or a path to a file with one genome name per line. Only these genomes will be "
                        "rendered, and they will be added to the animation in exactly the order they are "
                        "listed here, so this is also how you set the order of appearance. Every name must "
                        "occur in the pan-graph-db, and none may repeat. The default is every genome in the "
                        "pan-graph-db, in the order already stored there.")
    groupA.add_argument('--exclude-genomes', default=None, help="One or more genome names to leave out of "
                        "everything that follows, as a comma-separated list or a path to a file with one "
                        "genome name per line. This is applied after --genomes-of-interest, so the two work "
                        "together: you can ask for a set of genomes in a given order, and still drop a few of "
                        "them without editing that list.")
    groupA.add_argument('--max-num-genomes-to-render', default=None, type=int, help="Render only this many "
                        "genomes instead of every genome that is available. The animation then grows from 1 "
                        "genome up to this number, following the same order of appearance. Since a full "
                        "render can take a while, this is a convenient way to see what your output will look "
                        "like with a handful of genomes first.")

    groupB = parser.add_argument_group('INSET', "An optional second, magnified panel following one locus.")
    groupB.add_argument('--inset-flanks', default=None, metavar='LEFT_NODE,RIGHT_NODE', help="Two node ids "
                        "(as they appear in the pan-graph-db) flanking the locus to magnify in an inset "
                        "panel. Both must be present in every frame you render (i.e. conserved/backbone "
                        "nodes). Omit to render the whole graph only, with no inset.")
    groupB.add_argument('--inset-aspect', default=1.436, type=float, help="Width-to-height ratio of the "
                        "inset panel's frame (not necessarily of the locus drawn inside it). Default 1.436 "
                        "suits a tall, narrow locus (a 'spire'); a wide, short locus (e.g. a broad "
                        "hypervariable region) wants a larger value, closer to 4.")

    groupB.add_argument('--inset-graph-nodes-cover-window-dynamically', default=False, action='store_true',
                        help="Spread the inset's nodes across the panel's full width in EVERY frame, rather "
                        "than at one spacing pinned to the finished graph. By default a subgraph that grows "
                        "as genomes arrive starts out small, nested against the left edge of the panel, and "
                        "expands rightwards until the last genome fills the panel exactly. Pass this and the "
                        "spacing between nodes is instead solved fresh for each frame, so the two flanking "
                        "nodes sit at the panel's edges from the very first frame and what grows between them "
                        "is the structure alone. The levels keep their pinned height either way, so the "
                        "subgraph still gains height as it gains genomes.")

    groupD = parser.add_argument_group('GEOMETRY', "How the graph is drawn. Defaults are reasonable for most "
                                       "pangenomes; adjust if your graph looks too cramped or too sparse.")
    groupD.add_argument('--resolution', default='3840x2160', help="Output video resolution as WIDTHxHEIGHT "
                        "(default: 3840x2160). The panel layout is tuned for 16:9.")
    groupD.add_argument('--rotation-offset-deg', default=0.0, type=float, help="Rotates the whole main panel "
                        "by this many degrees, purely cosmetic (default: 0.0).")
    groupD.add_argument('--outer-radius-fraction', default=0.97, type=float, help="Radius the FINAL frame's "
                        "hole-plus-tracks-plus-graph is designed to reach exactly, as a fraction of the graph "
                        "panel's own half-size (default: 0.97, which leaves just enough margin that strokes "
                        "at the very edge don't get clipped). In other words, how much of the panel the whole "
                        "drawing fills. The height of the genome-tracks band has no separate knob of its own: "
                        "given this target, the hole (--inner-radius-fraction) and the graph's share (below), "
                        "each genome's track height is SOLVED FOR so the three always sum to exactly this "
                        "radius, with no leftover whitespace and nothing pushed past the edge, regardless of "
                        "how many genomes are being rendered. The panel itself is also sized to claim the most "
                        "space it can next to the text/inset column at your --resolution, so raising this is "
                        "what actually shrinks the remaining margin.")
    groupD.add_argument('--inner-radius-fraction', default=0.15, type=float, help="Radius of the empty hole "
                        "at the very center of the main panel, as a fraction of --outer-radius-fraction "
                        "(default: 0.15).")
    groupD.add_argument('--graph-height-fraction', default=0.2, type=float, help="How far the graph (the "
                        "backbone ring plus its spires) reaches beyond the tracks, as a fraction of the TOTAL "
                        "tracks height, which is to say of all genomes' rings stacked together. The default "
                        "of 0.2 means the graph reaches one-fifth as far beyond the tracks as the tracks "
                        "themselves reach beyond the hole, keeping the tracks the dominant visual element. "
                        "Together with --outer-radius-fraction and --inner-radius-fraction this fully "
                        "determines each track's height (see that flag's help), and raising it gives the "
                        "graph more relative room at the tracks' expense rather than at the panel's. The "
                        "spire-climb-per-step is then sized from the FINAL frame's real tallest spire so it "
                        "lands exactly on that reach.")
    groupD.add_argument('--backbone-gap-px', default=None, type=float, help="Fixed radial gap between the "
                        "genome-tracks stack and the backbone ring, in pixels (default: half of one track's "
                        "own height). This is the only distance between the two, and it does not grow with "
                        "the backbone's own length, which is what makes the backbone hug the tracks at every "
                        "frame.")
    groupD.add_argument('--track-height-px', default=None, type=float, help="Pixels per y (track/spire) step "
                        "in the main panel, which is to say how far a spire climbs outward from the backbone "
                        "per insertion. Setting this replaces the default derivation, which instead sizes the "
                        "step from the FINAL frame's real tallest spire so it lands exactly on "
                        "--graph-height-fraction's share of the panel. Use it when you want an exact, "
                        "resolution-independent pixel value, at the risk of over- or under-shooting the panel "
                        "edge.")
    groupD.add_argument('--supersample', default=2, type=int, help="Render this many times larger than the "
                        "final resolution and downsample, for antialiasing (default: 2; raise for smoother "
                        "lines at the cost of render time).")

    groupCosmetics = parser.add_argument_group('COSMETICS', "Node/edge ink for the main graph, the inset, and "
                                               "the genome tracks. Every default here is calculated from the "
                                               "actual data (genome count, panel size, zoom level, and so on) "
                                               "and is printed back under a 'COSMETICS' section, along with "
                                               "the exact flag to use, right before rendering starts. Since "
                                               "no formula can look right for every combination of genome "
                                               "count and locus shape, tune from what is printed there rather "
                                               "than guessing blind.")
    groupCosmetics.add_argument('--main-graph-node-size', default=None, type=float, help="Main-panel node "
                                "circle radius, in pixels. Default is derived from --resolution and how "
                                "densely packed the final frame's backbone is.")
    groupCosmetics.add_argument('--main-graph-node-color', default=None, help="Main-panel node fill color as "
                                "a hex code. By default every node keeps the color anvi'o's own state assigned "
                                "to its type, which is what makes core, accessory and singleton clusters tell "
                                "each other apart. Passing this OVERRIDES all of them to a single color.")
    groupCosmetics.add_argument('--main-graph-new-node-effect', default='none',
                                choices=['none'] + sorted(NEW_NODE_EFFECTS), help=f"Mark the nodes that arrive "
                                f"with each genome, in the main panel, with a brief effect, which makes what a "
                                f"genome actually contributed easy to follow. This marks only nodes that are "
                                f"genuinely NEW to the graph; --inset-graph-new-node-effect deliberately casts "
                                f"a wider net, see its own help. 'droplets' sends a filled copy of every new "
                                f"node breaking outwards to {NEW_NODE_EFFECTS['droplets'].max_scale:g}x its own "
                                f"size and fading as it spreads. The first genome is never marked, "
                                f"since every node in it would be. How long an effect runs for is a property of "
                                f"the effect itself, given as a share of --seconds-per-genome and printed back "
                                f"under 'ASSEMBLING VIDEO'. Default: none. Note that an effect makes every frame "
                                f"of a genome's turn on screen differ from the last, so both the render time and "
                                f"the size of the MP4 go up.")
    groupCosmetics.add_argument('--main-graph-nodes-fly-in', default=False, action='store_true', help="Have "
                                "the nodes a genome brings with it FLY IN to their places, instead of simply "
                                "being there in the next frame. Each new node comes in from off-canvas along "
                                "its own fixed bearing, oversized and shrinking as it goes, and easing to a "
                                "stop at exactly its true size once it lands. This makes what a genome actually "
                                "contributed impossible to miss and where in the graph it landed easy to "
                                "follow, and arriving large is what makes it work at all on a dense pangenome, "
                                "where a node resolves to a pixel or two across and could not otherwise be "
                                "followed on its way in. Only genuinely NEW nodes fly; everything already on "
                                "screen stays put, "
                                "and edges are drawn in full throughout, so the graph visibly holds a place "
                                "open for whatever is on its way in. The first genome never flies, since every "
                                "node in it would be. This is an alternative to --main-graph-new-node-effect "
                                "rather than a companion to it (asking for both is an error, since the effect "
                                "would fire where the node is going while the node is still on its way there), "
                                "though --inset-graph-new-node-effect is unaffected. Default: off. Note that "
                                "this makes every frame of a genome's turn on screen differ from the last, so "
                                "both the render time and the size of the MP4 go up; `--num-threads` is the "
                                "answer to most of that.")
    groupCosmetics.add_argument('--main-graph-node-edge-width', default=None, type=float, help="Main-panel "
                                "node outline stroke width, in pixels. Default: one third of "
                                "--main-graph-node-size.")
    groupCosmetics.add_argument('--main-graph-node-edge-color', default='#000000', help="Main-panel node "
                                "outline color as a hex code (default: '#000000', black).")
    groupCosmetics.add_argument('--main-graph-edge-width', default=None, type=float, help="Main-panel graph "
                                "edge (the lines connecting gene-cluster nodes) line width, in pixels. "
                                "Default is derived the same way as --main-graph-node-size.")
    groupCosmetics.add_argument('--main-graph-edge-color', default='#3C3C3C', help="Main-panel graph edge "
                                "color as a hex code (default: '#3C3C3C', dark gray).")
    groupCosmetics.add_argument('--main-graph-edge-opacity', default=0.63, type=float, help="Main-panel graph "
                                "edge opacity, 0-1 (default: 0.63).")
    groupCosmetics.add_argument('--inset-graph-level-height', default=None, type=float, help="How far apart "
                                "the inset panel's y levels sit, in pixels. A subgraph has as many levels as "
                                "its topology needs, and the two axes of the panel are scaled independently, "
                                "so by default the levels are spread across the panel's full height however "
                                "many of them there are. On a large subgraph that can still leave them close "
                                "enough together to bury the structure the panel exists to show, which is what "
                                "this is for: raise it to pull the levels apart, at the cost of the tallest of "
                                "them reaching past the top of the panel.")
    groupCosmetics.add_argument('--inset-graph-node-size', default=None, type=float, help="Inset-panel node "
                                "circle radius, in pixels. Default is derived from how zoomed-in the inset "
                                "is (wider gaps between backbone positions get bigger nodes).")
    groupCosmetics.add_argument('--inset-graph-node-color', default=None, help="Inset-panel node fill color "
                                "as a hex code. Works exactly like --main-graph-node-color, and is separate "
                                "from it so the magnified panel can carry the type colors while the main panel "
                                "goes flat, or the other way around.")
    groupCosmetics.add_argument('--inset-graph-new-node-effect', default='none',
                                choices=['none'] + sorted(NEW_NODE_EFFECTS), help="The same effects as "
                                "--main-graph-new-node-effect, in the magnified panel, but triggered on a "
                                "WIDER set of nodes than that flag's name suggests, so please read on. The "
                                "main panel marks only nodes that are genuinely new. The inset marks every "
                                "NON-CORE node the arriving genome puts a gene into, whether that node is new "
                                "or was already on screen. A genome dropping a gene into a synteny gene "
                                "cluster that several other genomes already occupy has changed that cluster, "
                                "and at this magnification that is worth watching, while in the main panel the "
                                "same rule would mark hundreds of nodes per genome and read as noise. Core "
                                "nodes are never marked either way: 'core' means every genome in the graph has "
                                "a gene there, so every genome would light up every core node, which says "
                                "nothing about any of them. Separate from --main-graph-new-node-effect so the "
                                "two can be asked for independently, though the timing is shared. Default: "
                                "none.")
    groupCosmetics.add_argument('--inset-graph-node-edge-width', default=None, type=float, help="Inset-panel "
                                "node outline stroke width, in pixels. Default: one sixth of "
                                "--inset-graph-node-size (thinner than the main panel's, since inset nodes "
                                "are already much larger).")
    groupCosmetics.add_argument('--inset-graph-node-edge-color', default='#000000', help="Inset-panel node "
                                "outline color as a hex code (default: '#000000', black).")
    groupCosmetics.add_argument('--inset-graph-edge-width', default=None, type=float, help="Inset-panel graph "
                                "edge line width, in pixels. Default is derived the same way as "
                                "--inset-graph-node-size.")
    groupCosmetics.add_argument('--inset-graph-edge-color', default='#3C3C3C', help="Inset-panel graph edge "
                                "color as a hex code (default: '#3C3C3C', dark gray).")
    groupCosmetics.add_argument('--inset-graph-edge-opacity', default=0.55, type=float, help="Inset-panel "
                                "graph edge opacity, 0-1 (default: 0.55).")
    groupCosmetics.add_argument('--genome-track-line-width', default=None, type=float, help="Genome-track "
                                "synteny-line width, in pixels. Default is derived from each track's own "
                                "(fixed) height.")
    groupCosmetics.add_argument('--genome-track-line-color', default=None, help="Genome-track synteny-line "
                                "color as a hex code. By default each genome keeps the color anvi'o's own "
                                "state assigned to it. Passing this OVERRIDES every genome to a single color.")
    groupCosmetics.add_argument('--genome-track-line-opacity', default=0.75, type=float, help="Genome-track "
                                "synteny-line opacity, 0-1 (default: 0.75).")
    groupCosmetics.add_argument('--genome-track-line-background', default=None, help="Genome-track background "
                                "band color as a hex code. By default the background color stored in the "
                                "pan-graph-db's own state is used (usually a near-white gray).")

    groupE = parser.add_argument_group('TIMING', "Pacing of the animation.")
    groupE.add_argument('--fps', default=30, type=int, help="Frames per second (default: 30).")
    groupE.add_argument('--hold-first-seconds', default=1.5, type=float, help="How long the first genome is "
                        "held before the animation starts moving (default: 1.5).")
    groupE.add_argument('--seconds-per-genome', default=0.45, type=float, help="How long each genome's state "
                        "is held before dissolving to the next (default: 0.45).")
    groupE.add_argument('--dissolve-seconds', default=0.2, type=float, help="Cross-dissolve duration between "
                        "consecutive genome states (default: 0.2).")
    groupE.add_argument('--hold-last-seconds', default=4.0, type=float, help="How long the finished graph is "
                        "held at the end (default: 4.0).")

    groupF = parser.add_argument_group('TEXT', "The title block above the genome counter: a small kicker "
                                       "line, an italic title, and an optional subtitle. For instance, 'The "
                                       "pangenome graph for', then 'Undatipelagibacter', then '(formerly, "
                                       "SAR11 1a.3.VI)'.")
    groupF.add_argument('--title-kicker', default='The pangenome graph for', help="Small muted line above "
                        "the title (default: 'The pangenome graph for'). Pass an empty string to hide it.")
    groupF.add_argument('--title', default=None, help="The main title line, set in italic (e.g. a genus "
                        "name). Defaults to the project name stored in the pan-graph-db.")
    groupF.add_argument('--subtitle', default=None, help="An optional line below the title, in regular "
                        "weight (e.g. a former name). Omit to show no subtitle line at all.")
    groupF.add_argument('--inset-description', default=None, help="A short line of text shown in the gap "
                        "between the genome-counter progress bar and the inset panel, in the same font/color "
                        "as --subtitle. Only shown when --inset-flanks is given. Default: \"Subgraph between "
                        "LEFT_NODE and RIGHT_NODE detailed\", with the actual two --inset-flanks node ids "
                        "filled in. Wraps to fit the text column's width. Pass an empty string to hide it.")
    groupF.add_argument('--font', default='helvetica', choices=list(FONT_SETS.keys()), help="Typeface for "
                        "all on-screen text (default: helvetica).")

    groupPerf = parser.add_argument_group('PERFORMANCE', "How hard anvi'o works at getting this done.")
    groupPerf.add_argument(*anvio.A('num-threads'), **anvio.K('num-threads', {'help': "How many frames to "
                           "rasterize at the same time (default: 1). Drawing a frame is mostly Pillow's C "
                           "code, which lets go of Python's global interpreter lock while it works, so this "
                           "genuinely does speed things up, though sublinearly: expect something nearer 3x "
                           "from 8 threads than 8x. It is the biggest lever you have whenever an effect is "
                           "switched on, since an effect is what makes every single frame of every genome's "
                           "turn have to be drawn separately instead of held. There is no point going past "
                           "the number of CPUs on your machine, and anvi'o will say so if you do. Note that "
                           "this governs RASTERIZING frames and nothing else: the ffmpeg encode at the end is "
                           "already parallel of its own accord, since libx264 detects your cores and helps "
                           "itself to them. Anvi'o deliberately does not pass this number on to ffmpeg, "
                           "because capping the encoder at this flag's default of one thread measures several "
                           "times SLOWER than leaving it alone."}))

    groupG = parser.add_argument_group('OUTPUT', "Where results go.")
    groupG.add_argument(*anvio.A('output-file'), **anvio.K('output-file', {'required': True,
                        'help': "Path for the output MP4."}))
    groupG.add_argument('--background-color', default='#FFFFFF', help="Canvas background color (default: white).")
    groupG.add_argument('--keep-frames-dir', default=None, help="If given, individual frame PNGs are written "
                        "here and kept after the video is made (useful for pulling out a still frame). "
                        "Default: a temporary directory that is deleted when done.")
    groupG.add_argument('--no-export-stills', default=False, action='store_true', help="By default, this "
                        "program also writes '<output>-first-frame.{svg,png}' and "
                        "'<output>-last-frame.{svg,png}' next to the video. Pass this flag to skip that and "
                        "only write the MP4.")
    groupG.add_argument('--dry-run', default=False, action='store_true', help="Stop right after the scenes "
                        "are built and the first/last-frame stills are written, before rasterizing frames or "
                        "calling ffmpeg, which skips the slow part entirely. Handy while tuning the COSMETICS "
                        "flags: run with --dry-run, look at '<output>-last-frame.png', adjust, repeat, and "
                        "only drop --dry-run once it looks right. Incompatible with --no-export-stills, since "
                        "that would leave nothing to look at.")
    groupG.add_argument('--no-genome-tracks', default=False, action='store_true', help="Turn off the "
                        "concentric per-genome presence rings drawn between the empty center and the "
                        "backbone (one ring per genome currently on screen, showing which of its gene "
                        "clusters are present at each backbone position).")

    args = parser.get_args(parser, auto_fill_anvio_dbs=True)

    # what every COSMETICS flag falls back to when it is left alone, so `report_cosmetics` can
    # tell a value that was asked for from one that was worked out from the data. Read off the
    # parser rather than restated, so the two can never drift apart.
    args.cosmetic_defaults = {action.dest: action.default for action in groupCosmetics._group_actions}

    return args


if __name__ == '__main__':
    main()
