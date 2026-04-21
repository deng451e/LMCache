# Interface: RemoteTransferAdapter (renamed)

> **This file has been superseded.**  
> `RemoteTransferAdapter` has been renamed and redesigned as **`RemoteIOAdapter`**.  
> See [remote_io_adapter_interface.md](remote_io_adapter_interface.md) for the
> current interface.
>
> Key change: `RemoteIOAdapter` combines the lookup protocol (ZMQ
> `LookupRequest / UnpinRequest` fan-out) with RDMA data transfer, removing the
> need for `RemoteController` to act as a client-side lookup dispatcher.
