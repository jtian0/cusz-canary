from textual.app import App, ComposeResult
from textual.widgets import Static, RadioSet, RadioButton
from textual.containers import ScrollableContainer, Container
from textual.widgets import Button, Footer, Header, Static
from textual.widgets import Footer, Label, ListItem, ListView
from textual.widgets import RadioButton, RadioSet


class OptionPredictor(Static):
    def compose(self) -> ComposeResult:
        with RadioSet():
            yield RadioButton("Lorenzo", value="lrz")
            yield RadioButton("Lorenzo prototype", value="lrz-proto")
            yield RadioButton("Lorenzo ZigZag", value="lrz-zz")
            yield RadioButton("spline interpolation", value="spl")


class OptionHistogram(Static):
    def compose(self) -> ComposeResult:
        with RadioSet(
            id="radioset-histogram",
            classes="container",
        ):
            yield RadioButton("generic", name="hist-generic", value=True)
            yield RadioButton("sparsity-aware", name="hist-sp")


class OptionCodec1(Static):
    def compose(self) -> ComposeResult:
        with RadioSet(
            id="radioset-codec1",
            classes="container",
        ):
            yield RadioButton("Huffman", name="codec1-huffman", value=True)
            # yield RadioButton("Huffman, revisited", name="codec1-huffman-rev", disabled=True),
            yield RadioButton("FZGPU codec", name="codec1-fzgcodec")


class OptionMode(Static):
    def compose(self) -> ComposeResult:
        with RadioSet(
            id="radioset-mode",
            classes="container",
        ):
            yield RadioButton("ABS", name="mode-abs", value=True)
            yield RadioButton("REL", name="mode-rel")


class OptionPredictor(Static):
    def compose(self) -> ComposeResult:
        with RadioSet(
            id="radioset-predictor",
            classes="container",
        ):
            yield RadioButton(
                "Lorenzo", name="predictor-lrz", tooltip='"lrz"', value=True
            )
            yield RadioButton(
                "Lorenzo prototype", name="predictor-lrz-proto", tooltip='"lrz-proto"'
            )
            yield RadioButton(
                "Lorenzo ZigZag", name="predictor-lrz-zz", tooltip='"lrz-zz"'
            )
            yield RadioButton(
                "spline interpolation", name="predictor-spl", tooltip='"spl"'
            )

    def on_mount(self) -> None:
        self.query_one(RadioSet).focus()


class CommandBuilderApp(App):
    dark = False

    CSS = """
    .container {
        margin: 1 2;
    }
    .command {
        margin-top: 2;
        padding: 1;
        border: solid gray;
    }
    """

    @staticmethod
    def _remove_prefix(full: str, prefix: str):
        return full.replace(prefix, "")

    # Helper function to get the selected RadioButton's label in a RadioSet
    def get_selected_label(self, radio_set_id: str, prefix: str = ""):
        radio_set = self.query_one(f"#{radio_set_id}", RadioSet)
        for button in radio_set.query(RadioButton):
            if button.value:  # Checks if this RadioButton is selected
                cmd_component = (
                    button.name
                )  # Return the label of the selected RadioButton
                if prefix != "":
                    cmd_component = cmd_component.replace(prefix, "")
                return cmd_component
        # raise ValueError(f"{radio_set_id} does hot have a selected value.")
        return None

    # def compose(self) -> ComposeResult:
    #     yield Header()
    #     yield Footer()
    #     yield ScrollableContainer(OptionPredictor(), OptionMode(), OptionHistogram(), OptionHistogram())
    #     self.command_display = Static("# command builder\n", classes="command")
    #     yield self.command_display

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()

        yield ScrollableContainer(
            OptionMode(), OptionPredictor(), OptionHistogram(), OptionCodec1()
        )

        yield Static("select predictor", classes="container")
        yield OptionPredictor()

        yield Static("select mode", classes="container")
        yield OptionMode()

        yield Static("select codec1", classes="container")
        yield OptionCodec1()

        yield Static("select histogram", classes="container")
        yield OptionHistogram()

        # display the default command
        self.command_display = Static(
            "# command builder\ncusz -m <mode> --hist <hist> --predictor <predictor> --codec <codec>",
            classes="command",
        )
        yield self.command_display

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        predictor = self.get_selected_label("radioset-predictor", "predictor-")
        mode = self.get_selected_label("radioset-mode", "mode-")
        codec = self.get_selected_label("radioset-codec1", "codec1-")
        hist = self.get_selected_label("radioset-histogram", "hist-")

        # Format the command
        command = (
            f"cusz -m {mode} --hist {hist} --predictor {predictor} --codec {codec}"
        )
        self.command_display.update(f"# command builder\n{command}")


if __name__ == "__main__":
    app = CommandBuilderApp()
    app.run()
