from src.graph_loader import GraphLoader
from src.models import MaterialFact


class IngestionService:
    def __init__(self, graph_loader: GraphLoader):
        self.graph_loader = graph_loader

    def ingest_material_fact(self, fact: MaterialFact) -> None:
        self.graph_loader.load_article(fact.article)
        self.graph_loader.load_experiment(fact.experiment)
        self.graph_loader.load_method(fact.method)

        self.graph_loader.load_material(fact.material)
        self.graph_loader.load_property(fact.prop)
        self.graph_loader.load_kpi(fact.kpi)

        self.graph_loader.link_article_experiment(
            article=fact.article,
            experiment=fact.experiment,
        )

        self.graph_loader.link_experiment_method(
            experiment=fact.experiment,
            method=fact.method,
        )

        self.graph_loader.link_experiment_material(
            experiment=fact.experiment,
            material=fact.material,
        )

        self.graph_loader.link_material_property(
            material=fact.material,
            prop=fact.prop,
            value=fact.value,
            unit=fact.unit,
        )

        self.graph_loader.link_property_kpi(
            prop=fact.prop,
            kpi=fact.kpi,
        )
