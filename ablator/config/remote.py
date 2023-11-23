from pathlib import Path
from ablator.config.main import ConfigBase, configclass
from ablator.config.types import Derived, Stateless, Optional


def _repr(self):
    return (
        type(self).__name__
        + "("
        + ", ".join(
            [
                f"{k}='{v}'" if isinstance(v, str) else f"{k}={repr(v)}"
                for k, v in self.to_dict(_repr=True).items()
            ]
        )
        + ")"
    )


def _clean_key_pem(key_pem: str):
    return key_pem.replace("\\n", "\n").replace("\n", "\\n")


class SSHConfig:
    """
    The configuration for connecting to a remote storage via SSH.

    Parameters
    ----------
    host : str
        the host address of the server
    user : str
        the ssh user-name
    port : int
        the port running ssh
    key_pem : str | None, optional
        the raw string of the private key.
        Must be provided if `key_file` is unspecified, by default None
    key_file : Path | None, optional
        path to the key_file containing the private key.
        Must be provided if `key_pem` is unspecified, by default None

    Raises
    ------
    ValueError
        When both `key_pem` and `key_file` are unspecified or specified.
    """

    def __init__(
        self,
        host: str,
        user: str,
        port: int,
        key_pem: str | None = None,
        key_file: Path | None = None,
        **_,
    ):
        self.key_pem: str
        if not (key_file is None) ^ (key_pem is None):
            raise ValueError("Must only provide either `key_pem` or `key_file`.")
        if key_file is not None:
            self.key_pem = _clean_key_pem(key_file.read_text())
        elif key_pem is not None:
            self.key_pem = _clean_key_pem(key_pem)
        self.host: str = host
        self.user: str = user
        self.port: int = port

    # pylint: disable=useless-type-doc,useless-param-doc
    def to_dict(self, _repr: bool = False) -> dict[str, str]:
        """
        dictionary representation of the configuration
        that can be then parsed to be used for RClone.

        Parameters
        ----------
        _repr : bool
            Whether to return a dictionary representation that omits fixed
            internal values.

        Returns
        -------
        dict[str, str]
            the dictionary representation of the configuration.
        """
        _dict = {
            "host": self.host,
            "user": self.user,
            "port": str(self.port),
            "key_pem": self.key_pem,
            "key_use_agent": "False",
            "type": "sftp",
        }
        if _repr:
            del _dict["type"]
            del _dict["key_use_agent"]
        return _dict

    def __repr__(self) -> str:
        return _repr(self)


@configclass
class S3Config(ConfigBase):
    provider: str
    access_key_id: str
    secret_access_key: str
    region: str = ""
    endpoint: str = ""
    # `env_auth` get the access key and secret key from the environment
    # variables. Must set `access_key_id`  and `secret_access_key`
    # to empty.
    env_auth: str = "false"
    # `location_constraint` used only when creating buckets.
    # must remain empty otherwise.
    location_constraint: str = ""
    acl: str = "private"
    server_side_encryption: str = ""
    storage_class: str = ""
    type: str = "s3"


@configclass
class RemoteConfig(ConfigBase):
    """
    Remote configuration that is used to synchronize the experiment artifacts with
    a remote server or S3 cloud storage bucket. For details on the remote configuration
    please consult rmount. https://github.com/fostiropoulos/rmount

    Attributes
    ----------
    ssh: Stateless[Optional[SSHConfig]]
        The ssh configuration to use to connect to the remote storage
    s3: Stateless[Optional[S3Config]]
        The ssh configuration to use to connect to the remote storage
    remote_path: Derived[Path]
        The remote path to create a connection with and the `local_path`.
    local_path: Derived[Path]
        The local path to map to the `remote_path`

    Raises
    ------
    ValueError
        When both `ssh` and `s3` configurations are unspecified or specified.

    Examples
    --------
    The following example defines a Remote configuration for connecting to `localhost`
    via SSH. You will need to adapt it to your credentials.

    >>> RemoteConfig(
    ...     ssh={
    ...         "host": "127.0.0.1",
    ...         "user": "root",
    ...         "port": "22",
    ...         "key_file": Path.home() / ".ssh" / "id_rsa",
    ...      }
    ... )

    """

    ssh: Stateless[Optional[SSHConfig]]
    s3: Stateless[Optional[S3Config]]
    remote_path: Derived[str]
    local_path: Derived[str]

    # flake8: noqa: DOC101,DOC106,DOC103,DOC109
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (self.ssh is not None) ^ (self.s3 is None):
            raise ValueError("Can not specify both `ssh` and `s3` arguments.")

    def get_config(self):
        if self.ssh is not None:
            return self.ssh

        return self.s3
