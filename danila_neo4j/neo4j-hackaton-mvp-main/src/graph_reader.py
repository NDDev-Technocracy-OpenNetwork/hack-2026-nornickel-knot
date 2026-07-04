class GraphReader:
    def __init__(self, driver):
        self.driver = driver

    def find_materials_for_kpi(self, kpi_name: str) -> list[dict]:
        records, _, _ = self.driver.execute_query(
            """
            MATCH
                (a:Article)-[:DESCRIBES]->(e:Experiment)-[:TESTS]->(m:Material)-[r:HAS_PROPERTY]->(p:Property)-[:AFFECTS]->(k:KPI {name: $kpi_name})
            OPTIONAL MATCH
                (e)-[:USES]->(method:Method)
            RETURN
                a.title AS article,
                e.name AS experiment,
                method.name AS method,
                m.name AS material,
                p.name AS property,
                k.name AS kpi,
                r.value AS value,
                r.unit AS unit
            ORDER BY r.value DESC
            """,
            kpi_name=kpi_name,
        )

        result = []

        for record in records:
            data = record.data()
            data["evidence_path"] = (
                f"{data['article']} -> "
                f"{data['experiment']} -> "
                f"{data['material']} -> "
                f"{data['property']} -> "
                f"{data['kpi']}"
            )
            result.append(data)

        return result
