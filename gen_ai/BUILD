package(default_visibility = ["//visibility:public"])

py_library(
    name="my_lib",
    srcs=["my_lib.py"],
    deps=[
        "//gen_ai/models:simple_dense_1",
    ],
)

py_binary(
    name="my_exec",
    srcs=["my_exec.py"],
    deps=[
        ":my_lib",
    ],
)
