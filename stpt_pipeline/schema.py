from pathlib import Path

import voluptuous as vo

# TODO: make sure they are lists of floats
cof_dist_schema = vo.Schema(
    {vo.Required('cof_x'): list, vo.Required('cof_y'): list, vo.Required('tan'): list}
)

schema = vo.Schema(
    {
        vo.Required('root_dir'): vo.Coerce(Path),
        vo.Required('output_dir'): vo.Coerce(Path),
        vo.Required('flat_file'): vo.Coerce(Path),
        vo.Required('cof_dist'): cof_dist_schema,
    }
)
