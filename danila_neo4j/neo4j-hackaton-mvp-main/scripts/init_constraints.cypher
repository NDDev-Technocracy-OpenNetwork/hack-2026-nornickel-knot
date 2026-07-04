CREATE CONSTRAINT article_id_unique IF NOT EXISTS
FOR (a:Article)
REQUIRE a.article_id IS UNIQUE;

CREATE CONSTRAINT experiment_id_unique IF NOT EXISTS
FOR (e:Experiment)
REQUIRE e.experiment_id IS UNIQUE;

CREATE CONSTRAINT method_id_unique IF NOT EXISTS
FOR (m:Method)
REQUIRE m.method_id IS UNIQUE;

CREATE CONSTRAINT material_id_unique IF NOT EXISTS
FOR (m:Material)
REQUIRE m.material_id IS UNIQUE;

CREATE CONSTRAINT property_id_unique IF NOT EXISTS
FOR (p:Property)
REQUIRE p.property_id IS UNIQUE;

CREATE CONSTRAINT kpi_id_unique IF NOT EXISTS
FOR (k:KPI)
REQUIRE k.kpi_id IS UNIQUE;