from textual.app import App, ComposeResult
from textual.widgets import Static, RadioSet, RadioButton
from textual.containers import ScrollableContainer, Container, Horizontal
from textual.widgets import Button, Footer, Header, Static
from textual.widgets import Footer, Label, ListItem, ListView
from textual.widgets import RadioButton, RadioSet
from textual.widgets import DirectoryTree


class OptionHistogram(Static):
    def compose(self) -> ComposeResult:
        with RadioSet(id="radioset-histogram", classes="container"):
            yield RadioButton(
                "generic", name="hist-generic", tooltip='"generic"', value=True
            )
            yield RadioButton("sparsity", name="hist-default", tooltip='"default"')


class OptionCodec1(Static):
    def compose(self) -> ComposeResult:
        with RadioSet(id="radioset-codec1", classes="container"):
            yield RadioButton(
                "Huffman", name="codec1-huffman", tooltip='"huffman"', value=True
            )
            # yield RadioButton("Huffman, revisited", name="codec1-huffman-rev", disabled=True),
            yield RadioButton(
                "FZGPU codec", name="codec1-fzgcodec", tooltip='"fzgcodec"'
            )


class OptionMode(Static):
    def compose(self) -> ComposeResult:
        with RadioSet(id="radioset-mode", classes="container"):
            yield RadioButton("ABS", name="mode-abs", value=True)
            yield RadioButton("REL", name="mode-rel")


class OptionPredictor(Static):
    def compose(self) -> ComposeResult:
        with RadioSet(id="radioset-predictor", classes="container"):
            yield RadioButton(
                "Lorenzo", name="predictor-lrz", tooltip='"lrz"', value=True
            )
            yield RadioButton(
                "Lorenzo proto", name="predictor-lrz-proto", tooltip='"lrz-proto"'
            )
            yield RadioButton(
                "Lorenzo ZigZag", name="predictor-lrz-zz", tooltip='"lrz-zz"'
            )
            yield RadioButton("interpolation", name="predictor-spl", tooltip='"spl"')

    # def on_mount(self) -> None:
    # self.query_one(RadioSet).focus()


class DirectoryTreeApp(Static):
    def compose(self) -> ComposeResult:
        yield DirectoryTree("~")


class CommandBuilderApp(App):
    CSS_PATH = "cmdbuilder.tcss"
    TITLE = "SGCC"
    SUB_TITLE = "pSZ/cuSZ command builder"

    dark = False

    predictor = ""
    mode = ""
    codec = ""
    hist = ""

    @staticmethod
    def _remove_prefix(full: str, prefix: str):
        return full.replace(prefix, "")

    async def on_mount(self) -> None:
        self.update_command()

    def update_command(self):
        predictor = self.get_selected_radioset_value("radioset-predictor", "predictor-")
        mode = self.get_selected_radioset_value("radioset-mode", "mode-")
        codec = self.get_selected_radioset_value("radioset-codec1", "codec1-")
        hist = self.get_selected_radioset_value("radioset-histogram", "hist-")
        command = (
            f"cusz -m {mode} --hist {hist} --predictor {predictor} --codec {codec}"
        )
        self.command_display.update(f"{command}")

    def get_selected_radioset_value(self, radio_set_id: str, prefix: str = ""):
        radio_set = self.query_one(f"#{radio_set_id}", RadioSet)
        for button in radio_set.query(RadioButton):
            if button.value:  # Checks if this RadioButton is selected
                cmd_component = (
                    button.name
                )  # Return the label of the selected RadioButton
                if prefix != "":
                    cmd_component = cmd_component.replace(prefix, "")
                return cmd_component
        raise ValueError(f"{radio_set_id} does hot have a selected value.")
        return None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Footer()

        component_mode = OptionMode(classes="option-field")
        component_mode.border_title = "mode"

        component_predictor = OptionPredictor(classes="option-field")
        component_predictor.border_title = "predictor"

        component_histogram = OptionHistogram(classes="option-field")
        component_histogram.border_title = "histogram"

        component_codec1 = OptionCodec1(classes="option-field")
        component_codec1.border_title = "codec1"

        yield Horizontal(
            component_mode,
            component_predictor,
            component_histogram,
            component_codec1,
        )

        yield Horizontal(DirectoryTreeApp(), id="input-select")

        # placeholder command
        command = f"cusz -m {self.mode} --hist {self.hist} --predictor {self.predictor} --codec {self.codec}"
        self.command_display = Static(f"{command}", classes="command-view")
        self.command_display.border_title = "# command builder"

        yield Horizontal(self.command_display)
        # yield self.command_display

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        self.update_command()


if __name__ == "__main__":
    app = CommandBuilderApp()
    app.run()
