---
name: cpp-exercism-selftest-tmpdir
description: 在 C++ Exercism 练习中实现后需自建临时测试编译验证时，先在工作区内建临时目录并设置 TMPDIR，再用 -std=c++17
  -Wall -Wextra -Wpedantic -Werror 编译运行，最后清理临时文件。
---

## When to use
在 C++ 练习中已按 stub 头文件实现 .cpp 后，需要自建临时 main/自测文件编译验证；或编译时出现 `couldn't create cache file .../xcrun_db-... (errno=Operation not permitted)` 报错。

## Instructions
1. 先读 stub 头文件与 CMakeLists，确认命名空间、函数/类签名和返回类型，只改目标 .cpp（必要时补头文件），不要动测试文件。
2. 在工作区内建临时目录（如 `.scratch/`、`.selftest/`、`.tmp/`），把自测 main 写在其中，不要写进系统临时目录。
3. 编译前 `export TMPDIR="$PWD/<临时目录>/tmp"` 并 `mkdir -p`，再执行 `g++/c++/clang++ -std=c++17 -Wall -Wextra -Wpedantic -Werror -I. <目标>.cpp <自测>.cpp -o <临时目录>/run && <临时目录>/run`。
4. 若自测断言失败，先核对是自测期望写错还是实现错误，修正后重跑；不要为通过自测而放宽实现。
5. 验证通过后 `rm -rf` 本次创建的临时目录与自测文件，保留原有项目文件。

## Verification
编译退出码为 0 且自测程序输出通过信息（如 `all self-tests passed`）；`ls -a` 确认工作区只剩原有 CMakeLists、目标 .cpp/.h。

## Example
`mkdir -p .scratch/tmp && export TMPDIR="$PWD/.scratch/tmp" && c++ -std=c++17 -Wall -Wextra -Wpedantic -Werror -I. space_age.cpp .scratch/main.cpp -o .scratch/run && ./.scratch/run`，通过后 `rm -rf .scratch`。
