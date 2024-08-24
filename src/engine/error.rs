use thiserror::Error;
use ocl::Error as OclError;

#[derive(Error, Debug)]
pub enum RendererError {
    #[error("Renderer not initialized!")]
    DimensionsTooBigError,
    #[error("Size too big, dimensions don't fit in u16!")]
    RendererNotInitializedError,
    #[error("Failed to build kernel!")]
    KernelBuildError(OclError),
    #[error("Failed to create buffer!")]
    CreateBufferError(OclError),
    #[error("Failed to add arguments!")]
    AddArgumentsError(OclError),
    #[error("Failed to execute kernel!")]
    ExecuteKernelError(OclError),
    #[error("Failed to read buffer!")]
    ReadBufferError(OclError),
    #[error("Unknown renderer error!")]
    Unknown,
}
