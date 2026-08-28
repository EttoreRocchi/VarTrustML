"""
Caller configuration module for VarTrustML.

:class:`CallerConfig` holds the settings for variant caller comparison.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class CallerConfig:
    """Configuration for variant caller comparison.

    Attributes:
        caller_columns (List[str]): List of caller column names in the dataset.
        combinations (List[str]): List of logical combination expressions to evaluate.
    """

    caller_columns: List[str] = field(default_factory=list)
    combinations: List[str] = field(default_factory=list)

    @staticmethod
    def get_default_combinations(callers: List[str]) -> List[str]:
        """Generate default logical combinations of callers.

        Generates all pairwise AND/OR combinations, plus all-callers
        AND (unanimous consensus) and OR (any caller) if > 2 callers.

        Parameters
        ----------
        callers : list of str
            List of caller column names.

        Returns
        -------
        list of str
            List of combination expression strings.
        """
        if len(callers) < 2:
            return []

        combinations = []

        # Pairwise AND combinations
        for i, c1 in enumerate(callers):
            for c2 in callers[i + 1 :]:
                combinations.append(f"{c1} AND {c2}")

        # Pairwise OR combinations
        for i, c1 in enumerate(callers):
            for c2 in callers[i + 1 :]:
                combinations.append(f"{c1} OR {c2}")

        # All-callers combinations (only if > 2 callers)
        if len(callers) > 2:
            combinations.append(" AND ".join(callers))
            combinations.append(" OR ".join(callers))

        return combinations

    @classmethod
    def from_experiment_config(
        cls,
        caller_columns: List[str],
        caller_combinations: List[str],
        include_default_combinations: bool = True,
    ) -> "CallerConfig":
        """Create CallerConfig from experiment configuration parameters.

        Parameters
        ----------
        caller_columns : list of str
            List of caller column names.
        caller_combinations : list of str
            Custom combinations from user.
        include_default_combinations : bool
            Whether to include auto-generated combinations.

        Returns
        -------
        CallerConfig
            Configured instance with user and default combinations.
        """
        all_combinations = list(caller_combinations)  # Copy user combinations

        if include_default_combinations:
            default_combos = cls.get_default_combinations(caller_columns)
            # Add defaults that aren't already specified
            for combo in default_combos:
                if combo not in all_combinations:
                    all_combinations.append(combo)

        return cls(caller_columns=caller_columns, combinations=all_combinations)
