
use std::sync::Arc;

use smg::{
    app_context::AppContext,
    config::PolicyConfig,
    core::ConnectionMode,
    policies::PolicyFactory,
    routers::RouterTrait,
};

use super::{
    http::{
        vllm_pd_router::{VllmPDRouter, VllmPDConfig},
    },
};
use crate::{
    conf::RoutingModeExt,
};

pub struct RouterFactoryExt;

impl RouterFactoryExt {
    /// Create a router instance from application context
    pub async fn create_router(ctx: &Arc<AppContext>, routing_mode_ext: RoutingModeExt) -> Result<Box<dyn RouterTrait>, String> {
        // Check connection mode and route to appropriate implementation
        match ctx.router_config.connection_mode {
            ConnectionMode::Grpc { .. } => {
                // Route to gRPC implementation based on routing mode
                match routing_mode_ext {
                    RoutingModeExt::VllmPrefillDecode {
                        prefill_urls: _,
                        decode_urls: _,
                        prefill_policy: _,
                        decode_policy: _,
                        discovery_address: _,
                        enable_profiling: _,
                        profile_timeout_secs: _,
                        intra_node_data_parallel_size: _,
                    } => Err("vLLM PD mode requires HTTP connection_mode".to_string()),
                    _ => Err("TBD".to_string())
                }
            }
            ConnectionMode::Http => {
                // Route to HTTP implementation based on routing mode
                match routing_mode_ext {
                    RoutingModeExt::VllmPrefillDecode  {
                        prefill_urls,
                        decode_urls,
                        prefill_policy,
                        decode_policy,
                        discovery_address,
                        enable_profiling,
                        profile_timeout_secs,
                        intra_node_data_parallel_size,
                    } => {
                        tracing::info!("Creating VllmPDRouter with prefill_urls: {:?}, decode_urls: {:?}, discovery: {:?}",
                                      prefill_urls, decode_urls, discovery_address);
                        let vllm_pd_config = VllmPDConfig {
                            enable_profiling,
                            profile_timeout_secs,
                            intra_node_data_parallel_size,
                        };
                        Self::create_vllm_pd_router(
                            prefill_urls,
                            decode_urls,
                            discovery_address.clone(),
                            vllm_pd_config,
                            prefill_policy.as_ref(),
                            decode_policy.as_ref(),
                            &ctx.router_config.policy,
                            ctx,
                        )
                        .await
                    }
                    _ => Err("TBD".to_string())
                }
            }
        }

    }
    
    /// Create a vLLM PD router with service discovery and/or static URLs
    pub async fn create_vllm_pd_router(
        prefill_urls: Vec<(String, Option<u16>)>,
        decode_urls: Vec<String>,
        discovery_address: Option<String>,
        vllm_pd_config: VllmPDConfig,
        prefill_policy_config: Option<&PolicyConfig>,
        decode_policy_config: Option<&PolicyConfig>,
        main_policy_config: &PolicyConfig,
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        // Initialize policies in PolicyRegistry - use specific policies if provided, otherwise fall back to main policy
        let prefill_policy =
            PolicyFactory::create_from_config(prefill_policy_config.unwrap_or(main_policy_config));
        let decode_policy =
            PolicyFactory::create_from_config(decode_policy_config.unwrap_or(main_policy_config));

        // Set the prefill and decode policies in the registry
        ctx.policy_registry.set_prefill_policy(prefill_policy);
        ctx.policy_registry.set_decode_policy(decode_policy);

        // Create vLLM PD router with both static URLs and service discovery support
        if discovery_address.is_some() {
            tracing::info!(
                "Creating VllmPDRouter with service discovery at: {:?}",
                discovery_address
            );
        }
        if !prefill_urls.is_empty() || !decode_urls.is_empty() {
            tracing::info!(
                "Creating VllmPDRouter with static URLs - prefill: {:?}, decode: {:?}",
                prefill_urls,
                decode_urls
            );
        }

        let router = VllmPDRouter::new(
            prefill_urls.to_vec(),
            decode_urls.to_vec(),
            discovery_address,
            vllm_pd_config,
            ctx,
        )
        .await?;
        tracing::info!("VllmPDRouter instance created successfully");

        Ok(Box::new(router))
    }

}