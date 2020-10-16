py_binary(name="discretezoo_main",
          python_version="PY3",
          srcs_version="PY3",
          srcs=["discretezoo_main.py"],
          deps=[
              "//discretezoo/attack:attack_loop",
              "//discretezoo/attack:estimation",
              "//discretezoo/attack:importance",
              "//discretezoo/attack:sampling",
              "//discretezoo/loss:adversarial_hinge",
              "//discretezoo/loss:semantic_similarity", ":attack_setup"
          ])

py_library(name="attack_setup",
           srcs=["attack_setup.py"],
           srcs_version="PY3",
           deps=[
               "//discretezoo/loss:adversarial_hinge",
               "//discretezoo/loss:semantic_similarity"
           ])

py_test(
  name = "attack_setup_test",
  srcs = ["attack_setup_test.py"],
  srcs_version = "PY3",
  python_version = "PY3",
  deps = [
    ":attack_setup",
  ]
)