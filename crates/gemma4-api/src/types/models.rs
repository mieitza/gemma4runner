use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct ModelList { pub object: String, pub data: Vec<ModelObject> }

#[derive(Debug, Clone, Serialize)]
pub struct ModelObject { pub id: String, pub object: String, pub created: u64, pub owned_by: String }
