# Upstream attribution

LCD-YOLO is implemented as a source fork of Ultralytics rather than as a
runtime monkey patch. The bundled package reports Ultralytics version `8.4.33`.
Custom model parsing and checkpoint loading depend on the module registrations
in `ultralytics/nn/modules/__init__.py` and `ultralytics/nn/tasks.py`.

Upstream project: <https://github.com/ultralytics/ultralytics>

The upstream and derived source are distributed under AGPL-3.0. Existing
Ultralytics copyright and license headers are preserved.
