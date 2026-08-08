project(
  'KNLMeansVk',
  'cpp',
  default_options: ['cpp_std=c++20', 'buildtype=release'],
  license: 'GPL-3.0-or-later',
  version: '2.0',
)

version_parts = meson.project_version().split('.')

add_project_arguments('-DKNLMVK_MAJOR_VERSION=' + version_parts[0], language: 'cpp')
add_project_arguments('-DKNLMVK_MINOR_VERSION=' + version_parts[1], language: 'cpp')

cpp = meson.get_compiler('cpp')
py = import('python').find_installation(pure: false)

# VapourSynth headers (R80+, for VSVulkan4.h): an explicit path wins, the Python package
# provides them otherwise.
vs_include = get_option('vapoursynth_include')
if vs_include == ''
  r = run_command(
    py,
    '-c', 'import vapoursynth as vs; print(vs.get_include())',
    check: true,
  )
  vs_include = r.stdout().strip()
endif

# Only the Vulkan headers are needed: the plugin never links the loader, the VapourSynth
# core hands it every entry point already resolved.
deps = []
inc_dirs = [include_directories('.')]
inc_dirs += include_directories(vs_include)
vulkan_include = get_option('vulkan_include')
if vulkan_include != ''
  inc_dirs += include_directories(vulkan_include)
else
  vulkan_dep = dependency('vulkan', required: false)
  if vulkan_dep.found()
    deps += vulkan_dep.partial_dependency(compile_args: true, includes: true)
  elif not cpp.has_header('vulkan/vulkan.h')
    error('Vulkan headers not found; install them or pass -Dvulkan_include=path/to/include')
  endif
endif

link_args = []
if get_option('static') and host_machine.system() == 'linux'
  link_args += ['-static-libgcc', '-static-libstdc++']
endif

# The kernel is embedded with C23 #embed where the compiler has it; this generated raw
# string header is the fallback for everything else (MSVC, older gcc).
shader_header = custom_target(
  'shader_comp_h',
  input: 'lib/shader.comp',
  output: 'shader_comp.h',
  command: [py, files('lib/embed_shader.py'), '@INPUT@', '@OUTPUT@'],
)

library(
  'knlmvulkan',
  ['lib/knlmvulkan.cpp', shader_header],
  include_directories: inc_dirs,
  dependencies: deps,
  link_args: link_args,
  gnu_symbol_visibility: 'hidden',
  install: false,
)
