def settings():
    from gradio import Textbox

    from modules.shared import OptionInfo, opts

    placeholder = "\n".join(
        ["[", '\t"First Block Cache",', "\t0.6,", "\t0.4,", "\tfalse,", "\t0,", "]"]
    )

    opts.add_option(
        "bc_always",
        OptionInfo(
            default="",
            label="Always Enabled Settings",
            component=Textbox,
            component_args={
                "placeholder": placeholder,
                "max_lines": 7,
                "lines": 7,
            },
            section=("bc", "Block Cache"),
            category_id="sd",
        ).info(
            "<code>str</code>, <code>float</code>, <code>float</code>, <code>bool</code>, <code>int</code>"
        ),
    )

    opts.add_option(
        "bc_hires",
        OptionInfo(
            False,
            label="Enable during Hires. fix",
            section=("bc", "Block Cache"),
            category_id="sd",
        ),
    )
