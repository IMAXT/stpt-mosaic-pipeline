import voluptuous as vo

schema = vo.Schema(
    {
        vo.Required('root_dir'): str,
        vo.Required('output_dir'): str,
        vo.Required('flat_file'): str,
    }
)
