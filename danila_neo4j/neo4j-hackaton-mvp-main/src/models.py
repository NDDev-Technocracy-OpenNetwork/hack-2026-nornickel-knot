from pydantic import BaseModel


class Article(BaseModel):
    article_id: str
    title: str


class Experiment(BaseModel):
    experiment_id: str
    name: str


class Method(BaseModel):
    method_id: str
    name: str


class Material(BaseModel):
    material_id: str
    name: str


class KPI(BaseModel):
    kpi_id: str
    name: str


class Property(BaseModel):
    property_id: str
    name: str


class MaterialFact(BaseModel):
    article: Article
    experiment: Experiment
    method: Method
    material: Material
    prop: Property
    kpi: KPI
    value: float | None = None
    unit: str | None = None
