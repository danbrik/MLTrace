"""preserve existing warp output geometry

Revision ID: 0026_warp_output_shape_mode
Revises: 0025_heatmap_range_runs
Create Date: 2026-06-22 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = "0026_warp_output_shape_mode"
down_revision: str | None = "0025_heatmap_range_runs"
branch_labels = None
depends_on = None


pipelines = sa.table(
    "preprocessing_pipelines",
    sa.column("id", sa.Integer()),
    sa.column("graph", sa.JSON()),
)


def _update_graphs(*, remove_mode: bool) -> None:
    bind = op.get_bind()
    for pipeline_id, graph in bind.execute(sa.select(pipelines.c.id, pipelines.c.graph)).fetchall():
        if not isinstance(graph, dict):
            continue
        changed = False
        for node in graph.get("nodes", []):
            if not isinstance(node, dict) or node.get("type") != "warp_perspective":
                continue
            config = node.setdefault("config", {})
            if not isinstance(config, dict):
                continue
            if remove_mode:
                changed = config.pop("output_shape_mode", None) is not None or changed
            elif "output_shape_mode" not in config:
                config["output_shape_mode"] = "manual"
                changed = True
        if changed:
            bind.execute(
                sa.update(pipelines).where(pipelines.c.id == pipeline_id).values(graph=graph)
            )


def upgrade() -> None:
    _update_graphs(remove_mode=False)


def downgrade() -> None:
    _update_graphs(remove_mode=True)
