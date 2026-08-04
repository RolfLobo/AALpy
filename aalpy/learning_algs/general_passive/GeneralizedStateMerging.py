# Core implementation of the Generalized State Merging (GSM) algorithm: a red-blue
# state-merging framework used to passively learn deterministic, nondeterministic and
# stochastic automata from data.
import functools
from collections import deque
from collections.abc import Callable
from typing import Any

from aalpy.base import Automaton
from aalpy.learning_algs.general_passive.GsmNode import GsmNode, OutputBehavior, TransitionBehavior, TransitionInfo, \
    OutputBehaviorRange, TransitionBehaviorRange, DataFormat, intersection_iterator, unknown_output, detect_data_format
from aalpy.learning_algs.general_passive.ScoreFunctionsGSM import ScoreCalculation, hoeffding_compatibility


# TODO add option for making checking of futures and partition non mutual exclusive?
#  Easiest done by adding a new method / field to ScoreCalculation

class Partitioning:
    """Represents the tentative result of merging a blue node into a red node, plus the resulting node mapping."""

    def __init__(self, red: GsmNode, blue: GsmNode) -> None:
        """
        Create a (not yet scored) partitioning for the merge of blue into red.

        :param GsmNode red: Red (already accepted) node the merge targets.
        :param GsmNode blue: Blue (candidate) node being merged.
        """
        self.red: GsmNode = red
        self.blue: GsmNode = blue
        self.score = False
        self.red_mapping: dict[GsmNode, GsmNode] = dict()
        self.full_mapping: dict[GsmNode, GsmNode] = dict()


class Instrumentation:
    """Base class for hooks that observe/report on the progress of GeneralizedStateMerging.run."""

    def __init__(self) -> None:
        """
        Create an instrumentation instance. No state by default.
        """
        pass

    def reset(self, gsm: 'GeneralizedStateMerging') -> None:
        """
        Called once at the start of a learning run.

        :param GeneralizedStateMerging gsm: The GSM instance being run.
        """
        pass

    def pta_construction_done(self, root: GsmNode) -> None:
        """
        Called after the initial PTA has been constructed.

        :param GsmNode root: Root node of the constructed PTA.
        """
        pass

    def log_promote(self, node: GsmNode) -> None:
        """
        Called whenever a blue node is promoted to red.

        :param GsmNode node: The promoted node.
        """
        pass

    def log_merge(self, part: Partitioning) -> None:
        """
        Called whenever a merge is performed.

        :param Partitioning part: The partitioning describing the performed merge.
        """
        pass

    def learning_done(self, root: GsmNode) -> None:
        """
        Called once learning has finished.

        :param GsmNode root: Root node of the learned model.
        """
        pass


class GeneralizedStateMerging:
    """Implements the red-blue state-merging framework used to passively learn automata from data."""

    def __init__(self, *,
                 output_behavior: OutputBehavior = "moore",
                 transition_behavior: TransitionBehavior = "deterministic",
                 score_calc: ScoreCalculation = None,
                 pta_preprocessing: Callable[[GsmNode], GsmNode] = None,
                 postprocessing: Callable[[GsmNode], GsmNode] = None,
                 compatibility_on_pta: bool = False,
                 compatibility_on_futures: bool = False,
                 node_order: Callable[[GsmNode, GsmNode], bool] = None,
                 consider_only_min_blue: bool = False,
                 depth_first: bool = False) -> None:
        """
        Configure a GeneralizedStateMerging instance.

        :param OutputBehavior output_behavior: Either "moore" or "mealy".
        :param TransitionBehavior transition_behavior: Either "deterministic", "nondeterministic" or "stochastic".
        :param ScoreCalculation score_calc: Local compatibility / global score calculation to use.
        :param Callable[[GsmNode], GsmNode] pta_preprocessing: Pre-processing function applied to the constructed PTA.
        :param Callable[[GsmNode], GsmNode] postprocessing: Post-processing function applied to the learned model.
        :param bool compatibility_on_pta: Whether compatibility is evaluated on the PTA instead of the current hypothesis.
        :param bool compatibility_on_futures: Whether compatibility is evaluated on futures instead of full partitions.
        :param Callable[[GsmNode, GsmNode], bool] node_order: Order in which merge candidates are considered.
        :param bool consider_only_min_blue: Whether to only consider the minimal blue node in each round.
        :param bool depth_first: Whether compatibility is checked depth-first instead of breadth-first.
        """

        if output_behavior not in OutputBehaviorRange:
            raise ValueError(f"invalid output behavior {output_behavior}. should be in {OutputBehaviorRange}")
        self.output_behavior: OutputBehavior = output_behavior
        if transition_behavior not in TransitionBehaviorRange:
            raise ValueError(f"invalid transition behavior {transition_behavior}. should be in {TransitionBehaviorRange}")
        self.transition_behavior: TransitionBehavior = transition_behavior

        if score_calc is None:
            if transition_behavior == "deterministic":
                score_calc = ScoreCalculation()
            elif transition_behavior == "nondeterministic" :
                raise ValueError("Missing score_calc for nondeterministic transition behavior. No default available.")
            elif transition_behavior == "stochastic" :
                score_calc = ScoreCalculation(hoeffding_compatibility(0.005, compatibility_on_pta))
        self.score_calc: ScoreCalculation = score_calc

        if node_order is None:
            self.node_order = GsmNode.default_order
        else:
            self.node_order = functools.cmp_to_key(lambda a, b: -1 if node_order(a, b) else 1)

        self.pta_preprocessing = pta_preprocessing or (lambda x: x)
        self.postprocessing = postprocessing or (lambda x: x)

        self.compatibility_on_pta = compatibility_on_pta
        self.compatibility_on_futures = compatibility_on_futures

        self.consider_only_min_blue = consider_only_min_blue
        self.depth_first = depth_first

    def compute_local_compatibility(self, a: GsmNode, b: GsmNode) -> bool:
        """
        Check whether two nodes are locally compatible, considering output/transition behavior and the score calculation.

        :param GsmNode a: First node.
        :param GsmNode b: Second node.
        :return bool: True if the nodes are locally compatible.
        """
        if self.output_behavior == "moore" and not GsmNode.moore_compatible(a, b):
            return False
        if self.transition_behavior == "deterministic" and not GsmNode.deterministic_compatible(a, b):
            return False
        return self.score_calc.local_compatibility(a, b)

    # TODO: make more generic by adding the option to use a different algorithm than red blue
    #  for selecting potential merge candidates. Maybe using inheritance with abstract `run`.
    def run(self, data: Any, convert: bool = True, instrumentation: Instrumentation | None = None,
            data_format: DataFormat | None = None) -> Automaton | GsmNode:
        """
        Run the state-merging algorithm on the provided data.

        :param Any data: Learning data, in one of the supported data formats (or already a GsmNode tree).
        :param bool convert: Whether to convert the resulting GsmNode tree into a concrete AALpy automaton.
        :param Instrumentation | None instrumentation: Instrumentation object used to report progress, defaults to a no-op instance.
        :param DataFormat | None data_format: Explicit data format of `data`, or None to auto-detect.
        :return Automaton | GsmNode: The learned automaton (if convert is True) or the raw GsmNode tree.
        """
        if instrumentation is None:
            instrumentation = Instrumentation()
        instrumentation.reset(self)

        if data_format is None:
            data_format = detect_data_format(data)
        if data_format == "labeled_sequences" and self.transition_behavior != "deterministic":
            raise ValueError("learning from labeled_sequences is not possible for nondeterministic systems")
        if data_format == "traces" and self.transition_behavior == "deterministic":
            print("learning deterministic systems from (output) traces only. this rarely makes sense. is `data_format` set correctly?")
        root = GsmNode.createPTA(data, self.output_behavior, data_format)

        root = self.pta_preprocessing(root)
        instrumentation.pta_construction_done(root)
        instrumentation.log_promote(root)

        if self.transition_behavior == "deterministic":
            if not root.is_deterministic():
                raise ValueError("required deterministic automaton but input data is nondeterministic")

        # sorted list of states already considered
        red_states = [root]

        partition_candidates: dict[tuple[GsmNode, GsmNode], Partitioning] = dict()
        while True:
            # sort states. states are always sorted using default order on original prefix
            if self.node_order is not GsmNode.default_order:
                red_states.sort(key=self.node_order)

            # get blue states
            blue_states = []
            for r in red_states:
                for _, _, t in r.transition_iterator():
                    c = t.target
                    if c in red_states:
                        continue
                    blue_states.append(c)
                    if self.consider_only_min_blue and self.node_order is GsmNode.default_order:
                        break

            # no blue states left -> done
            if len(blue_states) == 0:
                break
            if self.consider_only_min_blue: # does it make sense to check the score function here?
                blue_states = [min(blue_states, key=self.node_order)]
            if self.node_order is not GsmNode.default_order:
                blue_states.sort(key=self.node_order)

            # loop over blue states
            promotion = False
            for blue_state in blue_states:
                # FUTURE: Parallelize
                # FUTURE: Save partitions?

                # calculate partitions resulting from merges with red states if necessary
                current_candidates: dict[GsmNode, Partitioning] = dict()
                perfect_partitioning = None
                red_state = None
                for red_state in red_states:
                    partition = partition_candidates.get((red_state, blue_state))
                    if partition is None:
                        partition = self._partition_from_merge(red_state, blue_state)
                    if partition.score is True:
                        perfect_partitioning = partition
                        break
                    current_candidates[red_state] = partition
                assert red_state is not None

                # partition with perfect score found: don't consider anything else
                if perfect_partitioning:
                    partition_candidates = {(red_state, blue_state): perfect_partitioning}
                    break

                # no merge candidates for this blue state -> promote
                if all(part.score is False for part in current_candidates.values()):
                    red_states.append(blue_state)
                    instrumentation.log_promote(blue_state)
                    promotion = True
                    break

                # update tracking dict with new candidates
                new_candidates = (((red, blue_state), part) for red, part in current_candidates.items() if
                                  part.score is not False)
                partition_candidates.update(new_candidates)

            # a state was promoted -> don't clear candidates
            if promotion:
                continue

            # find best partitioning and clear candidates
            best_candidate = max(partition_candidates.values(), key=lambda part: part.score)
            for real_node, partition_node in best_candidate.red_mapping.items():
                real_node.transitions = partition_node.transitions
                real_node.predecessor = partition_node.predecessor
                real_node.prefix_access_pair = partition_node.prefix_access_pair
            instrumentation.log_merge(best_candidate)
            # FUTURE: optimizations for compatibility tests where merges can be orthogonal
            # FUTURE: caching for aggregating compatibility tests
            partition_candidates.clear()

        instrumentation.learning_done(root)

        root = self.postprocessing(root)
        if convert:
            root = root.to_automaton(self.output_behavior, self.transition_behavior)
        return root

    def _check_futures(self, red: GsmNode, blue: GsmNode) -> bool:
        """
        Check compatibility of the futures of two nodes, without constructing a full partition.

        :param GsmNode red: Red (already accepted) node.
        :param GsmNode blue: Blue (candidate) node.
        :return bool: True if all reachable node pairs are locally compatible.
        """
        q: deque[tuple[GsmNode, GsmNode]] = deque([(red, blue)])
        pop = q.pop if self.depth_first else q.popleft

        while len(q) != 0:
            red, blue = pop()

            if self.compute_local_compatibility(red, blue) is False:
                return False

            for in_sym, red_trans, blue_trans in intersection_iterator(red.transitions, blue.transitions):
                for out_sym, red_child, blue_child in intersection_iterator(red_trans, blue_trans):
                    if self.compatibility_on_pta:
                        if blue_child.original_count == 0 or red_child.original_count == 0:
                            continue
                        q.append((red_child.original_target, blue_child.original_target))
                    else:
                        q.append((red_child.target, blue_child.target))

        return True

    def _partition_from_merge(self, red: GsmNode, blue: GsmNode) -> Partitioning:
        """
        Compute the partitioning resulting from merging blue into red, including its score.

        Assumes that blue is a tree and red is not reachable from blue.

        :param GsmNode red: Red (already accepted) node the merge targets.
        :param GsmNode blue: Blue (candidate) node being merged.
        :return Partitioning: The resulting partitioning (with score False if incompatible).
        """
        partitioning = Partitioning(red, blue)

        self.score_calc.reset()

        if self.compatibility_on_futures:
            if self._check_futures(red, blue) is False:
                return partitioning

        # when compatibility is determined only by future and scores are disabled, we need not create partitions.
        if self.compatibility_on_futures and not self.score_calc.has_score_function():
            def update_partition(red_node: GsmNode, blue_node: GsmNode | None) -> GsmNode:
                return red_node
        else:
            def update_partition(red_node: GsmNode, blue_node: GsmNode | None) -> GsmNode:
                p = partitioning.full_mapping.get(red_node) # could check smaller .red_mapping?
                if p is None:
                    p = red_node.shallow_copy()
                    partitioning.full_mapping[red_node] = p
                    partitioning.red_mapping[red_node] = p
                if blue_node is not None:
                    partitioning.full_mapping[blue_node] = p
                return p

        # rewire the blue node's parent
        blue_parent = update_partition(blue.predecessor, None)
        blue_in_sym, blue_out_sym = blue.prefix_access_pair
        blue_parent.transitions[blue_in_sym][blue_out_sym].target = red

        partition = update_partition(red, None)
        if self.output_behavior == "moore":
            partition.resolve_unknown_prefix_output(blue_out_sym)

        # loop over implied merges
        q: deque[tuple[GsmNode, GsmNode]] = deque([(red, blue)])
        pop = q.pop if self.depth_first else q.popleft
        while len(q) != 0:
            red, blue = pop()
            partition = update_partition(red, blue)

            if not self.compatibility_on_futures:
                if self.compute_local_compatibility(partition, blue) is False:
                    return partitioning

            # create implied merges for all common successors
            for in_sym, blue_transitions in blue.transitions.items():
                partition_transitions = partition.transitions[in_sym]
                for out_sym, blue_transition in blue_transitions.items():
                    partition_transition = partition_transitions.get(out_sym)
                    # handle unknown output
                    if partition_transition is None and len(partition_transitions) != 0:
                        if out_sym is unknown_output:
                            assert len(partition_transitions) == 1
                            partition_transition = list(partition_transitions.values())[0]
                        if unknown_output in partition_transitions:
                            assert len(partition_transitions) == 1
                            partition_transition = partition_transitions.pop(unknown_output)
                            partition_transitions[out_sym] = partition_transition
                            # re-hook access pair
                            succ_part = update_partition(partition_transition.target, None)
                            if self.output_behavior == "moore" or succ_part.predecessor is red:
                                succ_part.resolve_unknown_prefix_output(out_sym)
                    # add pairs
                    if partition_transition is not None:
                        q.append((partition_transition.target, blue_transition.target))
                        partition_transition.count += blue_transition.count
                    else:
                        # blue child is blue after merging if there is a red state in blue's partition
                        partition_transition = TransitionInfo(blue_transition.target, blue_transition.count, None, 0)
                        partition_transitions[out_sym] = partition_transition
                        # update predecessor of blue child
                        blue_target_partition = update_partition(blue_transition.target, None)
                        blue_target_partition.predecessor = red

        partitioning.score = self.score_calc.score_function(partitioning.full_mapping)
        return partitioning


def run_GSM(data: list, *,
            output_behavior: OutputBehavior = "moore",
            transition_behavior: TransitionBehavior = "deterministic",
            score_calc: ScoreCalculation = None,
            pta_preprocessing: Callable[[GsmNode], GsmNode] = None,
            postprocessing: Callable[[GsmNode], GsmNode] = None,
            compatibility_on_pta: bool = False,
            compatibility_on_futures: bool = False,
            node_order: Callable[[GsmNode, GsmNode], bool] = None,
            consider_only_min_blue: bool = False,
            depth_first: bool = False,
            instrumentation: Instrumentation | None = None,
            convert: bool = True,
            data_format: DataFormat | None = None,
            ) -> Automaton | GsmNode:
    """
    Performs a state merging algorithm in the red-blue framework on provided data.

    :param list data: Data used for learning. Recorded behavior of the system.
    :param OutputBehavior output_behavior: Specifies whether outputs are emitted by states ("moore") or transitions ("mealy").
    :param TransitionBehavior transition_behavior: Either "deterministic", "nondeterministic" or "stochastic".
    :param ScoreCalculation score_calc: A ScoreCalculation object which determines how compatibility and merge scores are calculated.
    :param Callable[[GsmNode], GsmNode] pta_preprocessing: A pre-processing function applied to the PTA.
    :param Callable[[GsmNode], GsmNode] postprocessing: A postprocessing function applied to the learned automaton.
    :param bool compatibility_on_pta: Whether compatibility is evaluated on the PTA or the current hypothesis.
    :param bool compatibility_on_futures: Whether compatibility is evaluated using the futures of both states or all partition information.
    :param Callable[[GsmNode, GsmNode], bool] node_order: Order in which merge candidates are considered. Defaults to short-lex.
    :param bool consider_only_min_blue: Whether to consider merge candidates from all blue nodes or just a single.
    :param bool depth_first: Whether compatibility is checked depth- or breadth-first.
    :param Instrumentation | None instrumentation: Instrumentation object for reporting progress or debugging.
    :param bool convert: Whether to return a normal AALpy automaton type or a `GsmNode` object (internal representation).
    :param DataFormat | None data_format: Whether the input is given in the form of input-output traces or labeled input traces.
    :return Automaton | GsmNode: The learned automaton.
    """
    # instantiate gsm
    gsm = GeneralizedStateMerging(
        output_behavior=output_behavior,
        transition_behavior=transition_behavior,
        score_calc=score_calc,
        pta_preprocessing=pta_preprocessing,
        postprocessing=postprocessing,
        compatibility_on_pta=compatibility_on_pta,
        compatibility_on_futures=compatibility_on_futures,
        node_order=node_order,
        consider_only_min_blue=consider_only_min_blue,
        depth_first=depth_first,
    )

    # run the algorithm
    return gsm.run(data=data, instrumentation=instrumentation, convert=convert, data_format=data_format)
