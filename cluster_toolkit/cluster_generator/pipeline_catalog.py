from __future__ import annotations

from pathlib import Path

from .pipeline_models import RecipeGenerationProfile, TopologyTemplate


class PipelineCatalog:
    """Versioned topology and generation-profile lookup for the new pipeline."""

    def __init__(
        self,
        topologies: dict[str, TopologyTemplate],
        profiles: dict[str, RecipeGenerationProfile],
    ) -> None:
        if not topologies:
            raise ValueError("PipelineCatalog requires at least one topology")
        if not profiles:
            raise ValueError("PipelineCatalog requires at least one recipe profile")
        self._topologies = dict(topologies)
        self._profiles = dict(profiles)

        family_ids = {
            topology.family_id
            for topology in self._topologies.values()
            if topology.family_id is not None
        }

        for profile in self._profiles.values():
            unknown = sorted(set(profile.applies_to) - set(self._topologies))
            if unknown:
                raise ValueError(
                    f"Recipe profile {profile.profile_id} references unknown topologies: {unknown}"
                )
            unknown_families = sorted(
                set(profile.applies_to_families) - family_ids
            )
            if unknown_families and self._topologies:
                # Family profiles may be installed before a production run
                # materializes its first immutable topology snapshot.
                if profile.compiler != "atmospheric_linear":
                    raise ValueError(
                        f"Recipe profile {profile.profile_id} references unknown "
                        f"topology families: {unknown_families}"
                    )

    @classmethod
    def load(
        cls,
        topology_dir: str | Path,
        profile_dir: str | Path,
    ) -> "PipelineCatalog":
        topologies = cls._load_models(
            Path(topology_dir),
            TopologyTemplate,
            id_field="topology_id",
        )
        profiles = cls._load_models(
            Path(profile_dir),
            RecipeGenerationProfile,
            id_field="profile_id",
        )
        return cls(topologies=topologies, profiles=profiles)

    @staticmethod
    def _load_models(directory: Path, model_type, *, id_field: str) -> dict:
        if not directory.is_dir():
            raise FileNotFoundError(f"catalog directory does not exist: {directory}")

        loaded: dict[str, object] = {}
        for path in sorted(directory.rglob("*.json")):
            model = model_type.model_validate_json(path.read_text(encoding="utf-8"))
            model_id = getattr(model, id_field)
            if model_id in loaded:
                raise ValueError(f"duplicate {id_field} in catalog: {model_id}")
            loaded[model_id] = model
        if not loaded:
            raise ValueError(f"catalog directory contains no JSON definitions: {directory}")
        return loaded

    def topology(self, topology_id: str) -> TopologyTemplate:
        try:
            return self._topologies[topology_id]
        except KeyError as exc:
            raise KeyError(f"unknown topology_id: {topology_id}") from exc

    def profile(self, profile_id: str) -> RecipeGenerationProfile:
        try:
            return self._profiles[profile_id]
        except KeyError as exc:
            raise KeyError(f"unknown profile_id: {profile_id}") from exc

    @property
    def topology_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._topologies))

    @property
    def topologies(self) -> tuple[TopologyTemplate, ...]:
        """Return immutable catalog values in stable topology-ID order."""

        return tuple(self._topologies[item] for item in self.topology_ids)

    @property
    def profile_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._profiles))
