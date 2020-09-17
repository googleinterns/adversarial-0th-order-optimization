package(default_visibility = ["//discretezoo:__subpackages__"])

py_library(
  name = "estimation",
  srcs = ["estimation.py"],
  srcs_version = "PY3"
)

py_test(
  name = "estimation_test",
  srcs = ["estimation_test.py"],
  srcs_version = "PY3",
  python_version = "PY3",
  deps = [
    ":estimation",
  ]
)

py_library(
  name = "importance",
  srcs = ["importance.py"],
  srcs_version = "PY3"
)

py_test(
  name = "importance_test",
  srcs = ["importance_test.py"],
  srcs_version = "PY3",
  python_version = "PY3",
  deps = [
    ":importance",
  ]
)

py_library(
  name = "sampling",
  srcs = ["sampling.py"],
  srcs_version = "PY3"
)

py_test(
  name = "sampling_test",
  srcs = ["sampling_test.py"],
  srcs_version = "PY3",
  python_version = "PY3",
  deps = [
    ":sampling",
  ]
)

py_library(
  name = "attack_loop",
  srcs = ["attack_loop.py"],
  srcs_version = "PY3",
  deps = [
    ":estimation",
  ]
)

py_test(
  name = "attack_loop_test",
  srcs = ["attack_loop_test.py"],
  srcs_version = "PY3",
  python_version = "PY3",
  deps = [
    ":attack_loop",
  ]
)