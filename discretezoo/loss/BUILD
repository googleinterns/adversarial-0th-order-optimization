package(default_visibility = ["//discretezoo:__subpackages__"])

py_library(
  name = "adversarial_hinge",
  srcs = ["adversarial_hinge.py"],
  srcs_version = "PY3"
)

py_test(
  name = "adversarial_hinge_test",
  srcs = ["adversarial_hinge_test.py"],
  python_version = "PY3",
  srcs_version = "PY3",
  deps = [
    ":adversarial_hinge",]
)

py_library(
  name = "semantic_similarity",
  srcs = ["semantic_similarity.py"],
  srcs_version = "PY3",
  deps = [
    "//discretezoo:attack_setup"
  ]
)

py_test(
  name = "semantic_similarity_test",
  srcs = ["semantic_similarity_test.py"],
  python_version = "PY3",
  srcs_version = "PY3",
  deps = [
    ":semantic_similarity",
  ]
)
