// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Background index rebuild task.
//!
//! When alter_columns modifies a column with an index, the index is deleted and
//! a background rebuild task is scheduled. This module handles the execution
//! of those rebuild tasks.

use std::path::PathBuf;
use std::sync::Arc;

use lance_core::Result;
use lance_index::{IndexType, VectorIndexParams};
use lance_linalg::distance::MetricType;
use tracing::{error, info};

use crate::dataset::Dataset;
use crate::index::DatasetIndexExt;
use lance_table::format::IndexRebuildParams;

/// Semaphore to limit concurrent index rebuilds (avoid overwhelming resources)
use std::sync::Semaphore;
use once_cell::sync::Lazy;

static REBUILD_SEMAPHORE: Lazy<Semaphore> = Lazy::new(|| Semaphore::new(1));

/// Task information for rebuilding an index.
#[derive(Debug, Clone)]
pub struct IndexRebuildTask {
    /// Name of the index to rebuild.
    pub index_name: String,
    /// Path to the file containing rebuild parameters.
    pub params_path: PathBuf,
}

/// Execute the rebuild task in the background.
pub async fn execute_rebuild_task(
    dataset: &Dataset,
    task: &IndexRebuildTask,
) -> Result<()> {
    // Acquire semaphore to limit concurrent rebuilds
    let _permit = REBUILD_SEMAPHORE.acquire().await;
    
    info!("Starting background rebuild for index: {}", task.index_name);
    
    // Load rebuild parameters
    let params = load_rebuild_params(&task.params_path).await?;
    
    // Build index parameters
    let index_params = build_index_params(&params);
    
    // Create the index
    dataset
        .create_index(
            &[&params.column],
            IndexType::from_str(&params.index_type),
            None,  // name - will use default
            &index_params,
            true,  // replace if exists
        )
        .await?;
    
    // Cleanup temporary file
    cleanup_rebuild_params(&task.params_path).await?;
    
    info!("Completed rebuild for index: {}", task.index_name);
    Ok(())
}

/// Load rebuild parameters from disk.
pub async fn load_rebuild_params(path: &PathBuf) -> Result<IndexRebuildParams> {
    let data = tokio::fs::read(path).await?;
    let params: IndexRebuildParams = serde_json::from_slice(&data)
        .map_err(|e| lance_core::Error::invalid_input(format!(
            "Failed to parse rebuild params: {}",
            e
        )))?;
    Ok(params)
}

/// Save rebuild parameters to disk.
pub async fn save_rebuild_params(
    dataset: &Dataset,
    index_name: &str,
    params: &IndexRebuildParams,
) -> Result<PathBuf> {
    use std::path::Path;
    
    // Create directory for rebuild params
    let base_path = dataset.data_dir();
    let params_dir = base_path.join("_rebuild_params");
    tokio::fs::create_dir_all(&params_dir).await?;
    
    // Write params file
    let params_path = params_dir.join(format!("{}.json", index_name));
    let data = serde_json::to_vec_pretty(params)
        .map_err(|e| lance_core::Error::internal(format!(
            "Failed to serialize rebuild params: {}",
            e
        )))?;
    tokio::fs::write(&params_path, data).await?;
    
    Ok(params_path)
}

/// Cleanup temporary rebuild params file.
pub async fn cleanup_rebuild_params(path: &PathBuf) -> Result<()> {
    if path.exists() {
        tokio::fs::remove_file(path).await.ok();
    }
    Ok(())
}

/// Build IndexParams from rebuild params.
fn build_index_params(params: &IndexRebuildParams) -> VectorIndexParams {
    use lance_index::vector::ivf::IvfBuildParams;
    use lance_index::vector::pq::PQBuildParams;

    let distance_type = match params.distance_type.as_str() {
        "L2" => MetricType::L2,
        "Cosine" => MetricType::Cosine,
        "Dot" => MetricType::Dot,
        _ => MetricType::L2,
    };

    // Build stages based on index type
    let mut stages: Vec<lance_index::index::vector::StageParams> = vec![];

    // Add IVF stage if this is an IVF index
    if params.index_type.starts_with("IVF") {
        let num_partitions = params
            .target_partition_size
            .unwrap_or(256) as usize;
        stages.push(lance_index::index::vector::StageParams::Ivf(
            IvfBuildParams::new(num_partitions)
        ));
    }

    // Add compression stage
    if params.index_type.contains("PQ") {
        stages.push(lance_index::index::vector::StageParams::PQ(
            PQBuildParams {
                num_bits: params.num_bits.unwrap_or(8) as usize,
                num_sub_vectors: params.num_sub_vectors.unwrap_or(16) as usize,
                max_iters: 20,
                sample_rate: 256,
                kmeans_redos: 1,
            }
        ));
    }

    // Add HNSW stage if needed
    if params.index_type.contains("HNSW") {
        use lance_index::vector::hnsw::builder::HnswBuildParams;
        stages.push(lance_index::index::vector::StageParams::Hnsw(
            HnswBuildParams::default()
                .m(params.hnsw_m.unwrap_or(16) as u32)
                .ef_construction(params.hnsw_ef_construction.unwrap_or(200) as u32)
        ));
    }

    VectorIndexParams {
        stages,
        metric_type: distance_type,
        version: lance_index::IndexFileVersion::V3,
        skip_transpose: false,
        runtime_hints: std::collections::HashMap::new(),
    }
}

impl IndexType {
    pub fn from_str(s: &str) -> Self {
        match s {
            "IVF_PQ" => IndexType::IvfPq,
            "IVF_FLAT" => IndexType::IvfFlat,
            "IVF_SQ" => IndexType::IvfSq,
            "IVF_RQ" => IndexType::IvfRq,
            "IVF_HNSW_PQ" => IndexType::IvfHnswPq,
            "IVF_HNSW_FLAT" => IndexType::IvfHnswFlat,
            "IVF_HNSW_SQ" => IndexType::IvfHnswSq,
            "HNSW" => IndexType::Hnsw,
            "DISKANN" => IndexType::DiskAnn,
            "BTREE" => IndexType::BTree,
            _ => IndexType::IvfPq,
        }
    }
}

/// Spawn a background rebuild task.
pub fn spawn_rebuild_task(dataset: Dataset, task: IndexRebuildTask) {
    let dataset = Arc::new(dataset);
    
    tokio::spawn(async move {
        if let Err(e) = execute_rebuild_task(&dataset, &task).await {
            error!("Failed to rebuild index {}: {}", task.index_name, e);
        }
    });
}