from pydantic import BaseModel, Field, create_model
import litellm

class Node(BaseModel):
    id: str
    type: str = "Node"
    properties: dict = Field(default_factory=dict)

class Relationship(BaseModel):
    source: Node
    target: Node
    type: str
    properties: dict = Field(default_factory=dict)

GRAPH_EXTRACTION_PROMPT = [
    (
        "system",
        (
            "# Knowledge Graph Instructions\n"
            "## 1. Overview\n"
            "You are a top-tier algorithm designed for extracting information in structured "
            "formats to build a knowledge graph.\n"
            "Try to capture as much information from the text as possible without "
            "sacrificing accuracy. Do not add any information that is not explicitly "
            "mentioned in the text.\n"
            "- **Nodes** represent entities and concepts.\n"
            "- The aim is to achieve simplicity and clarity in the knowledge graph, making it\n"
            "accessible for a vast audience.\n"
            "## 2. Labeling Nodes\n"
            "- **Consistency**: Ensure you use available types for node labels.\n"
            "Ensure you use basic or elementary types for node labels.\n"
            "- For example, when you identify an entity representing a person, "
            "always label it as **'person'**. Avoid using more specific terms "
            "like 'mathematician' or 'scientist'."
            "- **Node IDs**: Never utilize integers as node IDs. Node IDs should be "
            "names or human-readable identifiers found in the text.\n"
            "- **Relationships** represent connections between entities or concepts.\n"
            "Ensure consistency and generality in relationship types when constructing "
            "knowledge graphs. Instead of using specific and momentary types "
            "such as 'BECAME_PROFESSOR', use more general and timeless relationship types "
            "like 'PROFESSOR'. Make sure to use general and timeless relationship types!\n"
            "## 3. Coreference Resolution\n"
            "- **Maintain Entity Consistency**: When extracting entities, it's vital to "
            "ensure consistency.\n"
            'If an entity, such as "John Doe", is mentioned multiple times in the text '
            'but is referred to by different names or pronouns (e.g., "Joe", "he"),'
            "always use the most complete identifier for that entity throughout the "
            'knowledge graph. In this example, use "John Doe" as the entity ID.\n'
            "Remember, the knowledge graph should be coherent and easily understandable, "
            "so maintaining consistency in entity references is crucial.\n"
            "## 4. Strict Compliance\n"
            "Adhere to the rules strictly. Non-compliance will result in termination."
        )
    ),
    (
        "user",
        (
            "Tip: Make sure to answer in the correct format and do "
            "not include any explanations. "
            "Use the given format to extract information from the "
            "following input: {input}"
        ),
    ),
]

def map_to_base_node(node) -> Node:
    properties = {}
    if hasattr(node, "properties") and node.properties:
        for p in node.properties:
            properties[format_property_key(p.key)] = p.value
    return Node(id=node.id, type=node.type, properties=properties)

def map_to_base_relationship(rel) -> Relationship:
    source = Node(id=rel.source_node_id, type=rel.source_node_type)
    target = Node(id=rel.target_node_id, type=rel.target_node_type)
    properties = {}
    if hasattr(rel, "properties") and rel.properties:
        for p in rel.properties:
            properties[format_property_key(p.key)] = p.value
    return Relationship(source=source, target=target, type=rel.type, properties=properties)

def _format_nodes(nodes: list[Node]) -> list[Node]:
    return [Node(
        id=el.id.title() if isinstance(el.id, str) else el.id,
        type=el.type.capitalize() if el.type else None,
        properties=el.properties,
    ) for el in nodes]

def _format_relationships(rels: list[Relationship]) -> list[Relationship]:
    return [Relationship(
        source=_format_nodes([el.source])[0],
        target=_format_nodes([el.target])[0],
        type=el.type.replace(" ", "_").upper(),
        properties=el.properties,
    ) for el in rels]

def format_property_key(s: str) -> str:
    words = s.split()
    if not words: return s
    return "".join([words[0].lower()] + [word.capitalize() for word in words[1:]])

class _Graph(BaseModel):
    nodes: list | None
    relationships: list | None

def _get_additional_info(input_type: str) -> str:
    if input_type not in ["node", "relationship", "property"]:
        raise ValueError("input_type must be 'node', 'relationship', or 'property'")
    if input_type == "node":
        return (
            "Ensure you use basic or elementary types for node labels.\n"
            "For example, when you identify an entity representing a person, "
            "always label it as **'Person'**. Avoid using more specific terms "
            "like 'Mathematician' or 'Scientist'"
        )
    elif input_type == "relationship":
        return (
            "Instead of using specific and momentary types such as "
            "'BECAME_PROFESSOR', use more general and timeless relationship types "
            "like 'PROFESSOR'. However, do not sacrifice any accuracy for generality"
        )
    return ""

def optional_enum_field(enum: list[str] | None, description: str = "", input_type: str = "node", **field_kwargs):
    if enum:
        return Field(
            ...,
            enum=enum,
            description=f"{description}. Available options are {enum}",
            **field_kwargs,
        )
    return Field(..., description=description + _get_additional_info(input_type), **field_kwargs)

def create_simple_model(
    node_labels: list[str] | None = None,
    rel_types: list[str] | None = None,
    node_properties: bool | list[str] = False,
    relationship_properties: bool | list[str] = False,
) -> type[_Graph]:
    node_fields = {
        "id": (str, Field(..., description="Name or human-readable unique identifier.")),
        "type": (str, optional_enum_field(
            node_labels,
            description="The type or label of the node.",
            input_type="node",
        ))
    }
    if node_properties:
        if isinstance(node_properties, list) and "id" in node_properties:
            raise ValueError("The node property 'id' is reserved and cannot be used.")
        
        class Property(BaseModel):
            """A single property consisting of key and value"""
            key: str = optional_enum_field(
                [] if node_properties is True else node_properties,
                description="Property key.",
                input_type="property",
            )
            value: str = Field(..., description="value")

        node_fields["properties"] = (list[Property] | None, Field(None, description="List of node properties"))
    
    relationship_fields = {
        "source_node_id": (str, Field(..., description="Name or human-readable unique identifier of source node")),
        "source_node_type": (str, optional_enum_field(
            node_labels,
            description="The type or label of the source node.",
            input_type="node"
        )),
        "target_node_id": (str, Field(..., description="Name or human-readable unique identifier of target node")),
        "target_node_type": (str, optional_enum_field(
            node_labels,
            description="The type or label of the target node.",
            input_type="node"
        )),
        "type": (str, optional_enum_field(
            rel_types,
            description="The type of the relationship.",
            input_type="relationship"
        ))
    }
    if relationship_properties:
        if isinstance(relationship_properties, list) and "id" in relationship_properties:
            raise ValueError("The relationship property 'id' is reserved and cannot be used.")

        class RelationshipProperty(BaseModel):
            """A single property consisting of key and value"""
            key: str = optional_enum_field(
                [] if relationship_properties is True else relationship_properties,
                description="Property key.",
                input_type="property",
            )
            value: str = Field(..., description="value")

        relationship_fields["properties"] = (
            list[RelationshipProperty] | None,
            Field(None, description="List of relationship properties"),
        )

    class DynamicGraph(_Graph):
        """Represents a graph document consisting of nodes and relationships."""
        nodes: list[create_model("SimpleNode", **node_fields)] | None = Field(description="List of nodes")  # type: ignore
        relationships: list[create_model("SimpleRelationship", **relationship_fields)] | None = Field(description="List of relationships")  # type: ignore

    return DynamicGraph

async def extract_graph(
    text: str,
    model: str = "gpt-4o-mini",
    allowed_nodes: list[str] = [],
    allowed_relationships: list[str] = [],
    node_properties: bool | list[str] = False,
    relationship_properties: bool | list[str] = False,
    **kwargs,
) -> tuple[list[Node], list[Relationship]]:
    schema = create_simple_model(allowed_nodes, allowed_relationships, node_properties, relationship_properties)
    messages = [{"role": m[0], "content": m[1].format(input=text)} for m in GRAPH_EXTRACTION_PROMPT]
    response = await litellm.acompletion(model, messages, response_format=schema, **kwargs)
    parsed_schema: _Graph = schema.model_validate_json(response.choices[0].message.content)
    return (
        _format_nodes([
                map_to_base_node(node)
                for node in parsed_schema.nodes
                if node.id
            ]
            if parsed_schema.nodes else []),
        _format_relationships([
                map_to_base_relationship(rel)
                for rel in parsed_schema.relationships
                if rel.type and rel.source_node_id and rel.target_node_id
            ]
            if parsed_schema.relationships else [])
    )
