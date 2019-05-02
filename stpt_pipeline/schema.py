from pathlib import Path

import voluptuous as vo

schema = vo.Schema(
    {
        vo.Required('root_dir'): vo.Coerce(Path),
        vo.Required('output_dir'): vo.Coerce(Path),
        vo.Required('flat_file'): vo.Coerce(Path),
    }
)
