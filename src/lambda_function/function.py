from __future__ import annotations

import html as _html
import re
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
from hashlib import sha256
from logging import INFO, getLogger
from os import environ
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional, Union, cast

import backoff
import boto3
import orjson
import requests
from aws_error_utils import aws_error_matches
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError
from bs4 import BeautifulSoup, NavigableString, Tag
from fastapi import (
    FastAPI,
    Depends,
    Header,
    HTTPException,
    status,
    Response,
    Request,
    UploadFile,
)
from fastapi.responses import (
    RedirectResponse,
    HTMLResponse,
    ORJSONResponse,
    PlainTextResponse,
)
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from mangum import Mangum
from mangum.handlers import ALB, APIGateway, HTTPGateway, LambdaAtEdge
from mangum.types import LambdaContext, LambdaEvent
from passlib.hash import pbkdf2_sha512
from pydantic import BaseModel, Field, validator, root_validator

if TYPE_CHECKING:
    from mypy_boto3_lambda.client import LambdaClient
    from mypy_boto3_s3.client import S3Client
    from mypy_boto3_s3.type_defs import GetObjectOutputTypeDef
else:
    GetObjectOutputTypeDef = dict
    LambdaClient = object
    S3Client = object

getLogger().setLevel(environ.get("LOGGING_LEVEL") or INFO)

ACCEPT_HEADER = ", ".join(
    [
        "application/vnd.pypi.simple.v1+json",
        "application/vnd.pypi.simple.v1+html;q=0.2",
        "text/html;q=0.01",  # For legacy compatibility
    ]
)
API_GATEWAY_BASE_PATH = environ.get("API_GATEWAY_BASE_PATH") or "/"
APP = FastAPI()
BUCKET = environ["BUCKET"]
DATABASE: dict[str, bool] = None
DIST_EXTENSIONS = {
    ".egg": "application/octet-stream",
    ".exe": "application/vnd.microsoft.portable-executable",
    ".tar.bz2": "application/x-tar",
    ".tar.gz": "application/x-tar",
    ".whl": "application/octet-stream",
    ".zip": "application/zip",
}
HTTP_BASIC = HTTPBasic()
LAMBDA_CLIENT: LambdaClient = None
LATEST_RE = re.compile(r"latest")
MANGUM = Mangum(api_gateway_base_path=API_GATEWAY_BASE_PATH, app=APP)
NORMALIZE_NAME_RE = re.compile(r"[-_.]+")
PACKAGE_METADATA = dict(
    author="Author",
    author_email="Author-email",
    classifiers="Classifier",
    description="Description",
    description_content_type="Description-Content-Type",
    download_url="Download-URL",
    dynamic="Dynamic",
    home_page="Home-page",
    keywords="Keywords",
    license="License",
    maintainer="Maintainer",
    maintainer_email="Maintainer-email",
    metadata_version="Metadata-Version",
    name="Name",
    platform="Platform",
    project_urls="Project-URL",
    provides="Provides",
    provides_dist="Provides-Dist",
    provides_extras="Provides-Extra",
    obsoletes="Obsoletes",
    obsoletes_dist="Obsoletes-Dist",
    requires="Requires",
    requires_dist="Requires-Dist",
    requires_external="Requires-External",
    requires_python="Requires-Python",
    summary="Summary",
    supported_platform="Supported-Platform",
    version="Version",
)
MIRROR_INDEX_URL = environ.get("MIRROR_INDEX_URL") or "https://pypi.org/simple/"
if not MIRROR_INDEX_URL.endswith("/"):
    MIRROR_INDEX_URL += "/"
REPO_PREFIX = environ.get("REPO_BASE_PREFIX") or ""
if REPO_PREFIX and not REPO_PREFIX.endswith("/"):
    REPO_PREFIX += "/"
DATABASE_KEY = f"{REPO_PREFIX}database.json"
V1_JSON_KEY = "index.json"
PYPI_LEGACY_HTML_INDEX_KEY = f"{REPO_PREFIX}legacy.html"
PYPI_SIMPLE_V1_HTML_INDEX_KEY = f"{REPO_PREFIX}index.html"
PYPI_SIMPLE_V1_JSON_INDEX_KEY = f"{REPO_PREFIX}{V1_JSON_KEY}"
PROJECTS_PREFIX = f"{REPO_PREFIX}projects/"
USERS_PREFIX = f"{REPO_PREFIX}users/"
S3_CLIENT: S3Client = None

##############################
##          MODELS          ##
##############################


def orjson_dumps(obj: Any, *, default: Callable[[Any], Any], **kwargs) -> str:
    # orjson.dumps returns bytes, to match standard json.dumps we need to decode
    getLogger().info(f"KWARGS:{kwargs}")
    return orjson.dumps(obj, default=default, **kwargs).decode()


class PyPIMeta(BaseModel, json_dumps=orjson_dumps, json_loads=orjson.loads):
    api_version: str = Field(alias="api-version")

    @validator("api_version")
    def api_version_validator(cls, value: str) -> str:
        if value != "1.0":
            raise ValueError(f"Received an unknown 'api-version': {value}")
        return value

    def html(self) -> str:
        return f'<meta name="pypi:repository-version" content="{self.api_version}"/>'


class PyPIProjectListProject(
    BaseModel, json_dumps=orjson_dumps, json_loads=orjson.loads
):
    name: str

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, PyPIProjectListProject) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)

    @validator("name")
    def convert_name(cls, value: Union[str, NavigableString]) -> str:
        return str(value)

    def hmtl(self) -> str:
        return f'<a href="/{self.normalized_name}/">{self.name}</a>'

    @property
    def normalized_name(self) -> str:
        return normalize_name(self.name)


class PyPIProjectList(BaseModel, json_dumps=orjson_dumps, json_loads=orjson.loads):
    meta: PyPIMeta
    projects: list[PyPIProjectListProject]

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

    def html(self) -> str:
        return f'<!DOCTYPE html><html><head>{self.meta.html()}<title>Simple index</title></head><body>{"".join([project.hmtl() for project in sorted(self.projects, key=lambda v: v.normalized_name)])}</body></html>'

    @classmethod
    def parse_response(cls, response: requests.Response) -> PyPIProjectList:
        response.raise_for_status()
        project_list: PyPIProjectList = None
        content_type = response.headers["content-type"]
        if content_type == "application/vnd.pypi.simple.v1+json":
            project_list = PyPIProjectList.parse_raw(response.content)
        elif content_type in ("application/vnd.pypi.simple.v1+html", "text/html"):
            soup = BeautifulSoup(response.content, "html.parser")
            project_list = PyPIProjectList.parse_obj(
                dict(
                    meta={
                        "api-version": soup.head.meta.get("content") or "1.0"
                        if soup.head.meta.get("name") == "pypi:repository-version"
                        else "1.0"
                    },
                    projects=[
                        dict(name=cast(Tag, a).string) for a in soup.body.find_all("a")
                    ],
                )
            )
        else:
            raise ValueError(f"Unknown Content-Type: {content_type}")
        return project_list


class PyPIProjectFile(BaseModel, json_dumps=orjson_dumps, json_loads=orjson.loads):
    dist_info_metadata: Optional[Union[bool, dict[str, str]]] = Field(
        alias="dist-info-metadata"
    )
    filename: str
    gpg_sig: Optional[bool] = Field(alias="gpg-sig")
    hashes: dict[str, str] = dict()
    requires_python: Optional[str] = Field(alias="requires-python")
    url: str
    yanked: Optional[Union[bool, str]]

    @validator("filename")
    def convert_filename(cls, value: Union[str, NavigableString]) -> str:
        return str(value)

    def html(self) -> str:
        def choose_hash(hashes: dict[str, str]) -> str:
            hash = ""
            if hashes:
                if "sha256" in hashes:
                    hash = f'sha256={self.hashes["sha256"]}'
                else:
                    item = list(hashes.items())[0]
                    hash = f"{item[0]}={item[1]}"
            return hash

        if hash := choose_hash(self.hashes):
            html = f'<a href="{self.url}#{hash}"'
        else:
            html = f'<a href="{self.url}"'
        if self.dist_info_metadata is not None:
            if isinstance(self.dist_info_metadata, dict) and (
                hash := choose_hash(self.dist_info_metadata)
            ):
                html += f' data-dist-info-metadata="{hash}"'
            else:
                html += ' data-dist-info-metadata="true"'
        if self.gpg_sig is not None:
            html += f' data-gpg-sig="{"true" if self.gpg_sig else "false"}"'
        if self.requires_python is not None:
            html += f' data-requires-python="{_html.escape(self.requires_python)}"'
        if self.yanked is not None and self.yanked:
            html += f' data-yanked="{"true" if isinstance(self.yanked, bool) else _html.escape(self.yanked)}"'
        return html + f">{self.filename}</a><br />"

    @property
    def s3_metadata(self) -> dict[str, str]:
        metadata = dict(hashes=orjson.dumps(self.hashes).decode(), url=self.url)
        if self.dist_info_metadata is not None:
            metadata["dist-info-metadata"] = orjson.dumps(
                self.dist_info_metadata
            ).decode()
        if self.gpg_sig is not None:
            metadata["gpg-sig"] = self.gpg_sig
        if self.requires_python is not None:
            metadata["requires-python"] = self.requires_python
        if self.yanked is not None:
            metadata["yanked"] = orjson.dumps(self.yanked).decode()
        return metadata


class PyPIProjectDetail(BaseModel, json_dumps=orjson_dumps, json_loads=orjson.loads):
    files: list[PyPIProjectFile] = list()
    meta: PyPIMeta
    name: str

    @classmethod
    def get_remote_project_detail(cls, project: str) -> PyPIProjectDetail:
        try:
            return cls.parse_response(
                project,
                requests.get(
                    headers=dict(Accept=ACCEPT_HEADER),
                    url=f"{MIRROR_INDEX_URL}{project}/",
                ),
            )
        except requests.HTTPError as he:
            raise HTTPException(
                status_code=cast(requests.Response, he.response).status_code
            ) from he

    def html(self) -> str:
        return f'<!DOCTYPE html><html><head>{self.meta.html()}<title>Links for {self.name}</title></head><body><h1>Links for {self.name}</h1>{"".join([file.html() for file in sorted(self.files, key=lambda v: v.filename)])}</body></html>'

    @classmethod
    def parse_response(
        cls, name: str, response: requests.Response
    ) -> PyPIProjectDetail:
        response.raise_for_status()
        project_detail: PyPIProjectDetail = None
        content_type = response.headers["content-type"]
        if content_type == "application/vnd.pypi.simple.v1+json":
            project_detail = PyPIProjectDetail.parse_raw(response.content)
        elif content_type in ("application/vnd.pypi.simple.v1+html", "text/html"):
            soup = BeautifulSoup(response.content, "html.parser")

            def convert_a_tag(a: Tag) -> dict:
                href_components = a["href"].split("#")
                file = dict(filename=a.string, url=href_components[0])
                if len(href_components) > 1:
                    hash = href_components[1].split("=")
                    file["hashes"] = {hash[0]: hash[1]}
                if data_gpg_sig := a.get("data-gpg-sig"):
                    file["gpg-sig"] = data_gpg_sig == "true"
                if data_dist_info_metadata := a.get("data-dist-info-metadata"):
                    if data_dist_info_metadata == "true":
                        file["dist-info-metadata"] = True
                    else:
                        hash = data_dist_info_metadata.split("=")
                        file["dist-info-metadata"] = {hash[0]: hash[1]}
                if data_requires_python := a.get("data-requires-python"):
                    file["requires-python"] = _html.unescape(data_requires_python)
                if data_yanked := a.get("data-yanked"):
                    file["yanked"] = _html.unescape(data_yanked)
                return file

            project_detail = PyPIProjectDetail.parse_obj(
                dict(
                    files=[convert_a_tag(a) for a in soup.body.find_all("a")],
                    meta={
                        "api-version": soup.head.meta.get("content") or "1.0"
                        if soup.head.meta.get("name") == "pypi:repository-version"
                        else "1.0"
                    },
                    name=name,
                )
            )
        else:
            raise ValueError(f"Unknown Content-Type: {content_type}")
        return project_detail

    @classmethod
    def parse_get_object_type_def(
        cls, object_type_def: GetObjectOutputTypeDef
    ) -> PyPIProjectDetail:
        return PyPIProjectDetail.parse_raw(
            object_type_def["Body"].read(), content_type=object_type_def["ContentType"]
        )


class PyPIUser(BaseModel, json_dumps=orjson_dumps, json_loads=orjson.loads):
    hash: str
    upload: bool

    @classmethod
    def parse_get_object_type_def(
        cls, object_type_def: GetObjectOutputTypeDef
    ) -> PyPIUser:
        return PyPIUser.parse_raw(
            object_type_def["Body"].read(), content_type=object_type_def["ContentType"]
        )

    @root_validator(pre=True)
    def validate_and_hash(cls, values):
        if password := values.pop("password", None):
            values["hash"] = pbkdf2_sha512.hash(password)
        elif "hash" not in values:
            raise ValueError("Missing password")
        return values

    def verify(self, password: str) -> bool:
        return pbkdf2_sha512.verify(password, self.hash)


USERS_DATABASE: dict[str, PyPIUser] = dict()


##############################
##      HELPER FUNCTIONS    ##
##############################


def create_index(context: LambdaContext = None) -> None:
    getLogger().info("Indexing")
    getLogger().info(f"Retrieving remote index from {MIRROR_INDEX_URL}")
    project_list = PyPIProjectList.parse_response(
        requests.get(
            headers=dict(Accept=ACCEPT_HEADER),
            url=MIRROR_INDEX_URL,
        )
    )
    getLogger().info("Adding package repositories to the index")
    items = {
        object_summary.key[len(PROJECTS_PREFIX) :].split("/")[0]
        for object_summary in boto3.resource(
            "s3",
            config=BotocoreConfig(signature_version="s3v4"),
            region_name=environ["AWS_REGION"],
        )
        .Bucket(BUCKET)
        .objects.filter(Prefix=f"{PROJECTS_PREFIX}")
    }
    database = {project.normalized_name: False for project in project_list.projects} | {
        normalize_name(item): True for item in items
    }
    project_list.projects = sorted(
        list(
            {PyPIProjectListProject(name=item) for item in items}
            | set(project_list.projects)
        ),
        key=lambda v: v.name,
    )
    getLogger().info("Writing index files")
    html_body = project_list.html().encode()
    with ThreadPoolExecutor(max_workers=4) as executor:
        wait(
            [
                executor.submit(
                    s3_client().put_object,
                    Body=project_list.json(by_alias=True).encode(),
                    Bucket=BUCKET,
                    ContentType="application/vnd.pypi.simple.v1+json",
                    Key=PYPI_SIMPLE_V1_JSON_INDEX_KEY,
                ),
                executor.submit(
                    s3_client().put_object,
                    Body=html_body,
                    Bucket=BUCKET,
                    ContentType="text/html",
                    Key=PYPI_LEGACY_HTML_INDEX_KEY,
                ),
                executor.submit(
                    s3_client().put_object,
                    Body=html_body,
                    Bucket=BUCKET,
                    ContentType="application/vnd.pypi.simple.v1+html",
                    Key=PYPI_SIMPLE_V1_HTML_INDEX_KEY,
                ),
                executor.submit(
                    s3_client().put_object,
                    Body=orjson.dumps(database),
                    Bucket=BUCKET,
                    ContentType="application/json",
                    Key=DATABASE_KEY,
                ),
            ]
        )
    getLogger().info("Index created")
    restart_lambda_function(context)


def lambda_client() -> LambdaClient:
    global LAMBDA_CLIENT
    if not LAMBDA_CLIENT:
        LAMBDA_CLIENT = boto3.client("lambda", region_name=environ["AWS_REGION"])
    return LAMBDA_CLIENT


def load_database(reentry: bool = False) -> None:
    global DATABASE
    try:
        DATABASE = orjson.loads(
            boto3.resource(
                "s3",
                config=BotocoreConfig(signature_version="s3v4"),
                region_name=environ["AWS_REGION"],
            )
            .Object(BUCKET, DATABASE_KEY)
            .get()["Body"]
            .read()
        )
    except ClientError as ce:
        if not aws_error_matches(ce, "NoSuchKey") or reentry:
            raise ce
        create_index()
        load_database(reentry=True)


def normalize_name(name: str) -> str:
    return NORMALIZE_NAME_RE.sub("-", name).lower()


def restart_lambda_function(context: LambdaContext) -> None:
    if context:

        @backoff.on_exception(
            backoff.expo,
            ClientError,
            giveup=lambda e: not aws_error_matches(
                e, "ResourceConflictException", "TooManyRequestsException"
            ),
            max_time=context.get_remaining_time_in_millis() / 1000,
        )
        def _restart_lambda_function() -> None:
            lambda_client().update_function_configuration(
                Description=f"Reindexed - {datetime.now(timezone.utc).isoformat()}",
                FunctionName=context.function_name,
            )

        getLogger().warning(f"Restarting {context.function_name}")
        _restart_lambda_function()


def s3_client() -> S3Client:
    global S3_CLIENT
    if not S3_CLIENT:
        S3_CLIENT = boto3.client(
            "s3",
            config=BotocoreConfig(signature_version="s3v4"),
            region_name=environ["AWS_REGION"],
        )
    return S3_CLIENT


##############################
##           API            ##
##############################
class AuthorizedUser(NamedTuple):
    username: str
    upload: bool


async def authorize_user(
    credentials: HTTPBasicCredentials = Depends(HTTP_BASIC),
) -> AuthorizedUser:
    if not (user := USERS_DATABASE.get(credentials.username)):
        try:
            hashed_username = sha256(credentials.username.encode()).hexdigest()
            USERS_DATABASE[
                credentials.username
            ] = user = PyPIUser.parse_get_object_type_def(
                s3_client().get_object(
                    Bucket=BUCKET, Key=f"{USERS_PREFIX}{hashed_username}.json"
                )
            )
        except ClientError as ce:
            if not aws_error_matches(ce, "NoSuchKey"):
                raise ce
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
    if not user.verify(credentials.password):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
    return AuthorizedUser(username=credentials.username, upload=user.upload)


async def content_type_negotiation(accept: Optional[str] = Header(default="text/html")):
    content_types: list[tuple[str, str]] = list()
    for accept_option in accept.split(","):
        accept_option = accept_option.strip()
        accept_comps = accept_option.split(";")
        if len(accept_comps) == 1:
            content_types.append(("q=1.0", accept_comps[0].strip()))
        else:
            content_types.append((accept_comps[1].strip(), accept_comps[0].strip()))
    found_splat = False
    for content_type in sorted(content_types, reverse=True):
        if content_type[1] in (
            "application/vnd.pypi.simple.latest+json",
            "application/vnd.pypi.simple.v1+json",
            "application/vnd.pypi.simple.latest+html",
            "application/vnd.pypi.simple.v1+html",
            "text/html",
        ):
            return LATEST_RE.sub("v1", content_type[1])
        elif content_type[1] in ("*/*", "text/*"):
            found_splat = True
    if found_splat:
        return "text/html"
    raise HTTPException(
        detail=f"Unsuported media types: {accept}",
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    )


@APP.get("/simple/{project}/")
async def get_project_detail(
    project: str,
    content_type: str = Depends(content_type_negotiation),
    user: AuthorizedUser = Depends(authorize_user),
) -> Response:
    if (normalized_name := normalize_name(project)) != project:
        return RedirectResponse(
            f"/simple/{normalized_name}/",
            status_code=status.HTTP_308_PERMANENT_REDIRECT,
        )
    if (local := DATABASE.get(project)) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    project_detail: PyPIProjectDetail = None
    if local:
        getLogger().debug(
            f'Getting local project detail for "{project}", user "{user.username}", content type "{content_type}"'
        )
        params = dict(Bucket=BUCKET, Key=f"{PROJECTS_PREFIX}{project}/{V1_JSON_KEY}")
        print(params)
        try:
            project_detail = PyPIProjectDetail.parse_get_object_type_def(
                s3_client().get_object(**params)
            )
        except ClientError as ce:
            if not aws_error_matches(ce, "NoSuchKey"):
                raise ce
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from ce
    else:
        getLogger().debug(
            f'Getting remote project detail for "{project}", user "{user.username}", content type "{content_type}"'
        )
        project_detail = PyPIProjectDetail.get_remote_project_detail(project)
        for file in project_detail.files:
            file.url = f"/simple/{project}/{file.filename}"
    response: Response = None
    if content_type == "application/vnd.pypi.simple.v1+json":
        response = ORJSONResponse(
            content=project_detail.dict(by_alias=True, exclude_unset=True),
            media_type=content_type,
        )
    elif content_type in ("application/vnd.pypi.simple.v1+html", "text/html"):
        response = HTMLResponse(content=project_detail.html(), media_type=content_type)
    return response


@APP.get("/simple/{project}/{filename}")
async def get_project_file(
    project: str, filename: str, user: AuthorizedUser = Depends(authorize_user)
) -> RedirectResponse:
    if (normalized_name := normalize_name(project)) != project:
        return RedirectResponse(
            f"/simple/{normalized_name}/{filename}",
            status_code=status.HTTP_308_PERMANENT_REDIRECT,
        )
    if (local := DATABASE.get(project)) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    if local:
        getLogger().debug(
            f'Getting local project file "{filename}" for "{project}", user "{user.username}"'
        )
        params = dict(Bucket=BUCKET, Key=f"{PROJECTS_PREFIX}{project}/{filename}")
        try:
            s3_client().head_object(**params)
        except ClientError as ce:
            if not aws_error_matches(ce, "404"):
                raise ce
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        return RedirectResponse(
            s3_client().generate_presigned_url(
                ClientMethod="get_object",
                ExpiresIn=600,
                Params=params,
            )
        )
    else:
        getLogger().debug(
            f'Getting remote project file "{filename}" for "{project}", user "{user.username}"'
        )
        params = dict(Bucket=BUCKET, Key=f"{REPO_PREFIX}cache/{project}/{filename}")
        try:
            s3_client().head_object(**params)
        except ClientError as ce:
            if not aws_error_matches(ce, "404"):
                raise ce
            project_detail = PyPIProjectDetail.get_remote_project_detail(project)
            found = False
            for file in project_detail.files:
                if file.filename == filename:
                    found = True
                    break
            if not found:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
            getLogger().info(f"Pulling through {filename} from {project} at {file.url}")
            try:
                response = requests.get(file.url)
                response.raise_for_status()
                s3_client().put_object(
                    Body=response.content,
                    ContentType=response.headers["content-type"],
                    Metadata=file.s3_metadata,
                    **params,
                )
            except requests.HTTPError as he:
                raise HTTPException(
                    status_code=cast(requests.Response, he.response).status_code
                ) from he
        return RedirectResponse(
            s3_client().generate_presigned_url(
                ClientMethod="get_object",
                ExpiresIn=600,
                Params=params,
            )
        )


@APP.get("/simple/")
async def get_project_list(
    content_type: str = Depends(content_type_negotiation),
    user: AuthorizedUser = Depends(authorize_user),
) -> RedirectResponse:
    getLogger().debug(
        f'Getting project list, user "{user.username}", content type "{content_type}"'
    )
    key = PYPI_LEGACY_HTML_INDEX_KEY
    if content_type == "application/vnd.pypi.simple.v1+json":
        key = PYPI_SIMPLE_V1_JSON_INDEX_KEY
    elif content_type == "application/vnd.pypi.simple.v1+html":
        key = PYPI_SIMPLE_V1_HTML_INDEX_KEY
    return RedirectResponse(
        s3_client().generate_presigned_url(
            ClientMethod="get_object",
            ExpiresIn=600,
            Params=dict(Bucket=BUCKET, Key=key),
        )
    )


@APP.get("/simple/{project}")
async def redirect_project_detail(project: str) -> RedirectResponse:
    return RedirectResponse(
        f"/simple/{project}/", status_code=status.HTTP_308_PERMANENT_REDIRECT
    )


@APP.get("/simple")
async def redirect_project_list() -> RedirectResponse:
    return RedirectResponse("/simple/", status_code=status.HTTP_308_PERMANENT_REDIRECT)


@APP.post("/", status_code=status.HTTP_201_CREATED)
async def upload_project_file(
    request: Request, user: AuthorizedUser = Depends(authorize_user)
) -> Optional[Response]:
    if not user.upload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    form = await request.form()
    project = normalize_name(form["name"])
    project_file: UploadFile = form["content"]
    getLogger().info(f'Uploading {project_file.filename} to {form["name"]}')
    project_prefix = f"{PROJECTS_PREFIX}{project}/"
    project_file_params = dict(
        Bucket=BUCKET, Key=f"{project_prefix}{project_file.filename}"
    )
    project_v1_json_key = f"{project_prefix}{V1_JSON_KEY}"
    try:
        s3_client().head_object(**project_file_params)
    except ClientError as ce:
        if not aws_error_matches(ce, "404"):
            raise ce
    else:
        return PlainTextResponse(
            content=f"{project_file.filename} already exists",
            status_code=status.HTTP_409_CONFLICT,
        )
    content_type = "application/octet-stream"
    for ext, c_type in DIST_EXTENSIONS.items():
        if project_file.filename.endswith(ext):
            content_type = c_type
            break
    project_detail: PyPIProjectDetail = None
    reindex = False
    try:
        project_detail = PyPIProjectDetail.parse_get_object_type_def(
            s3_client().get_object(Bucket=BUCKET, Key=project_v1_json_key)
        )
    except ClientError as ce:
        if not aws_error_matches(ce, "NoSuchKey"):
            raise ce
        project_detail = PyPIProjectDetail(
            meta=PyPIMeta.parse_obj({"api-version": "1.0"}), name=form["name"]
        )
        reindex = True
    s3_client().put_object(
        Body=project_file.file, ContentType=content_type, **project_file_params
    )
    metadata: bytes = b""
    for key, value in form.multi_items():
        if (metatdata_name := PACKAGE_METADATA.get(key)) and value:
            if key == "description":
                value = value.replace("\n", "\n       |")
            metadata += f"{metatdata_name}: {value}\n".encode()
    s3_client().put_object(
        Body=metadata,
        Bucket=project_file_params["Bucket"],
        ContentType="text/plain",
        Key=project_file_params["Key"] + ".metadata",
    )
    project_file_obj = {
        "dist-info-metadata": dict(sha256=sha256(metadata).hexdigest()),
        "filename": project_file.filename,
        "hashes": {
            cast(str, key).split("_")[0]: value
            for key, value in form.items()
            if key in ("md5_digest", "sha256_digest") and value
        },
        "url": f"/simple/{project}/{project_file.filename}",
        "yanked": False,
    }
    if requires_python := form.get("requires_python"):
        project_file_obj["requires-python"] = requires_python
    if gpg_signature := form.get("gpg_signature"):
        project_file_obj["gpg-sig"] = True
        s3_client().put_object(
            Body=cast(UploadFile, gpg_signature).file,
            Bucket=BUCKET,
            ContentType="text/plain",
            Key=project_file_params["Key"] + ".asc",
        )
    project_detail.files.append(PyPIProjectFile.parse_obj(project_file_obj))
    s3_client().put_object(
        Body=project_detail.json(by_alias=True, exclude_none=True),
        Bucket=BUCKET,
        ContentType="application/vnd.pypi.simple.v1+json",
        Key=project_v1_json_key,
    )
    if reindex and "AWS_LAMBDA_FUNCTION_NAME" in environ:
        lambda_client().invoke(
            FunctionName=environ["AWS_LAMBDA_FUNCTION_NAME"],
            InvocationType="Event",
            Payload="reindex",
        )


##############################
##         USER MGT         ##
##############################


def put_user(
    username: str, password: str, context: LambdaContext, upload: bool = False
) -> None:
    getLogger().info(
        f'Adding/updating user {username}{" with upload priveleges" if upload else ""}'
    )
    hased_username = sha256(username.encode()).hexdigest()
    s3_client().put_object(
        Body=PyPIUser.parse_obj(dict(password=password, upload=upload)).json(),
        Bucket=BUCKET,
        ContentType="application/json",
        Key=f"{USERS_PREFIX}{hased_username}.json",
    )
    getLogger().info(
        f"Added/updated user {username}, restarting {context.function_name}"
    )
    restart_lambda_function(context)


def remove_user(username: str, context: LambdaContext) -> None:
    getLogger().info(f"Removing user {username}")
    hashed_username = sha256(username.encode()).hexdigest()
    try:
        s3_client().delete_object(
            Bucket=BUCKET, Key=f"{USERS_PREFIX}{hashed_username}.json"
        )
    except ClientError as ce:
        if not aws_error_matches(ce, "NoSuchKey"):
            raise ce
    else:
        getLogger().info(f"Removed user {username}, restarting {context.function_name}")
        restart_lambda_function(context)


##############################
##           MAIN           ##
##############################
load_database()


def handler(event: LambdaEvent, context: LambdaContext) -> Any:
    getLogger().debug(
        "Processing event:\n{}".format(
            orjson.dumps(event, option=orjson.OPT_INDENT_2).decode()
        )
    )
    if (
        APIGateway.infer(event, context, MANGUM.config)
        or ALB.infer(event, context, MANGUM.config)
        or HTTPGateway.infer(event, context, MANGUM.config)
        or LambdaAtEdge.infer(event, context, MANGUM.config)
    ):
        return MANGUM(event, context)
    elif "putUser" in event:
        username = cast(dict, event["putUser"])
        return put_user(
            username["username"],
            username["password"],
            context,
            username.get("upload", False),
        )
    elif "removeUser" in event:
        return remove_user(event["removeUser"], context)
    elif "reheat" in event:
        getLogger().info("Reheating")
    else:
        return create_index(context)
