"""Central product identity used by the app, packaging, and installer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AppIdentity:
    product_name: str
    application_name: str
    application_id: str
    organization_name: str
    organization_domain: str
    settings_directory_name: str
    settings_application_name: str
    mutex_prefix: str
    activation_event_prefix: str
    runtime_directory_name: str
    file_prefix: str
    executable_name: str
    installer_app_id: str


APP_IDENTITY = AppIdentity(
    product_name="Zumly Capture",
    application_name="ZumlyCapture",
    application_id="com.zumly.capture",
    organization_name="Zumly",
    organization_domain="zumly.app",
    settings_directory_name="Zumly Capture",
    settings_application_name="Zumly Capture",
    mutex_prefix=r"Local\ZumlyCapture.Tray",
    activation_event_prefix=r"Local\ZumlyCapture.Tray.Activate",
    runtime_directory_name="ZumlyCapture",
    file_prefix="zumly_capture",
    executable_name="ZumlyCapture",
    installer_app_id="{6F2A0B62-8A6B-4F16-96D5-0A2739C44D0B}",
)

PRODUCT_NAME = APP_IDENTITY.product_name
APPLICATION_NAME = APP_IDENTITY.application_name
APPLICATION_ID = APP_IDENTITY.application_id
ORGANIZATION_NAME = APP_IDENTITY.organization_name
ORGANIZATION_DOMAIN = APP_IDENTITY.organization_domain
SETTINGS_DIRECTORY_NAME = APP_IDENTITY.settings_directory_name
SETTINGS_APPLICATION_NAME = APP_IDENTITY.settings_application_name
MUTEX_PREFIX = APP_IDENTITY.mutex_prefix
ACTIVATION_EVENT_PREFIX = APP_IDENTITY.activation_event_prefix
RUNTIME_DIRECTORY_NAME = APP_IDENTITY.runtime_directory_name
FILE_PREFIX = APP_IDENTITY.file_prefix
EXECUTABLE_NAME = APP_IDENTITY.executable_name
INSTALLER_APP_ID = APP_IDENTITY.installer_app_id
