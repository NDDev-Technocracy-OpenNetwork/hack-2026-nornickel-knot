from src.models import Article, Experiment, Method, Material, KPI, Property


class GraphLoader:
    def __init__(self, driver):
        self.driver = driver

    def load_material(self, material: Material) -> None:
        self.driver.execute_query(
            """
            MERGE (m:Material {material_id: $material_id})
            SET m.name = $name
            """,
            material_id=material.material_id,
            name=material.name,
        )

    def load_property(self, prop: Property) -> None:
        self.driver.execute_query(
            """
            MERGE (p:Property {property_id: $property_id})
            SET p.name = $name
            """,
            property_id=prop.property_id,
            name=prop.name,
        )

    def load_kpi(self, kpi: KPI) -> None:
        self.driver.execute_query(
            """
            MERGE (k:KPI {kpi_id: $kpi_id})
            SET k.name = $name
            """,
            kpi_id=kpi.kpi_id,
            name=kpi.name,
        )

    def link_material_property(
        self,
        material: Material,
        prop: Property,
        value: float | None = None,
        unit: str | None = None,
    ) -> None:
        self.driver.execute_query(
            """
            MATCH (m:Material {material_id: $material_id})
            MATCH (p:Property {property_id: $property_id})
            MERGE (m)-[r:HAS_PROPERTY]->(p)
            SET r.value = $value,
                r.unit = $unit  
            """,
            material_id=material.material_id,
            property_id=prop.property_id,
            value=value,
            unit=unit,
        )

    def link_property_kpi(self, prop: Property, kpi: KPI) -> None:
        self.driver.execute_query(
            """
            MATCH (p:Property {property_id: $property_id})
            MATCH (k:KPI {kpi_id: $kpi_id})
            MERGE (p)-[:AFFECTS]->(k)
            """,
            property_id=prop.property_id,
            kpi_id=kpi.kpi_id,
        )

    def load_article(self, article: Article) -> None:
        self.driver.execute_query(
            """
            MERGE (a:Article {article_id: $article_id})
            SET a.title = $title
            """,
            article_id=article.article_id,
            title=article.title,
        )

    def load_experiment(self, experiment: Experiment) -> None:
        self.driver.execute_query(
            """
            MERGE (e:Experiment {experiment_id: $experiment_id})
            SET e.name = $name
            """,
            experiment_id=experiment.experiment_id,
            name=experiment.name,
        )

    def load_method(self, method: Method) -> None:
        self.driver.execute_query(
            """
            MERGE (method:Method {method_id: $method_id})
            SET method.name = $name
            """,
            method_id=method.method_id,
            name=method.name,
        )

    def link_article_experiment(self, article: Article, experiment: Experiment) -> None:
        self.driver.execute_query(
            """
            MATCH (a:Article {article_id: $article_id})
            MATCH (e:Experiment {experiment_id: $experiment_id})
            MERGE (a)-[:DESCRIBES]->(e)
            """,
            article_id=article.article_id,
            experiment_id=experiment.experiment_id,
        )

    def link_experiment_method(self, experiment: Experiment, method: Method) -> None:
        self.driver.execute_query(
            """
            MATCH (e:Experiment {experiment_id: $experiment_id})
            MATCH (method:Method {method_id: $method_id})
            MERGE (e)-[:USES]->(method)
            """,
            experiment_id=experiment.experiment_id,
            method_id=method.method_id,
        )

    def link_experiment_material(
        self, experiment: Experiment, material: Material
    ) -> None:
        self.driver.execute_query(
            """
            MATCH (e:Experiment {experiment_id: $experiment_id})
            MATCH (m:Material {material_id: $material_id})
            MERGE (e)-[:TESTS]->(m)
            """,
            experiment_id=experiment.experiment_id,
            material_id=material.material_id,
        )
