"""Animated solution for Problem 01: one Transformer projection cost."""

from manim import *


BG = "#08111F"
WHITE_TEXT = "#F3F7FC"
MUTED = "#AFC2D6"
BLUE = "#58A6D8"
BLUE_DARK = "#173A56"
GREEN = "#56D3A3"
GREEN_DARK = "#123C32"
ORANGE = "#F2A65A"
ORANGE_DARK = "#4B301B"
PURPLE = "#B58BDD"
PURPLE_DARK = "#322342"
YELLOW = "#FFD166"
RED = "#F17C87"


class TransformerProjectionCost(MovingCameraScene):
    """Visual derivation of parameters, HBM bytes, and FLOPs."""

    def construct(self):
        self.camera.background_color = BG
        self.show_title_and_question()
        self.show_shape_mechanics()
        self.show_parameter_count()
        self.show_hbm_bytes()
        self.show_flop_count()
        self.show_summary_and_boundary()

    def header(self, title, subtitle=None):
        heading = Text(title, font_size=34, weight=BOLD, color=WHITE_TEXT).to_edge(UP, buff=0.28)
        items = [heading]
        if subtitle:
            sub = Text(subtitle, font_size=19, color=MUTED).next_to(heading, DOWN, buff=0.12)
            items.append(sub)
        return VGroup(*items)

    def pill(self, text, color=BLUE, fill=BLUE_DARK, font_size=21):
        label = Text(text, font_size=font_size, color=WHITE_TEXT)
        box = RoundedRectangle(
            corner_radius=0.12,
            width=label.width + 0.42,
            height=label.height + 0.25,
            stroke_color=color,
            stroke_width=2,
            fill_color=fill,
            fill_opacity=1,
        )
        label.move_to(box)
        return VGroup(box, label)

    def matrix_card(self, name, shape, rows, cols, color, fill, width=3.0, height=2.2):
        outer = RoundedRectangle(
            corner_radius=0.16,
            width=width,
            height=height,
            stroke_color=color,
            stroke_width=3,
            fill_color=fill,
            fill_opacity=0.96,
        )
        title = MathTex(name, font_size=38, color=WHITE_TEXT).next_to(outer.get_top(), DOWN, buff=0.22)
        grid_width = width * 0.72
        grid_height = height * 0.36
        cells = VGroup()
        for row in range(rows):
            for col in range(cols):
                cell = Rectangle(
                    width=grid_width / cols,
                    height=grid_height / rows,
                    stroke_color=color,
                    stroke_width=1,
                    fill_color=color,
                    fill_opacity=0.13,
                )
                cell.move_to(
                    outer.get_center()
                    + LEFT * grid_width / 2
                    + RIGHT * (col + 0.5) * grid_width / cols
                    + UP * 0.10
                    + UP * grid_height / 2
                    + DOWN * (row + 0.5) * grid_height / rows
                )
                cells.add(cell)
        shape_label = MathTex(shape, font_size=29, color=color).next_to(outer.get_bottom(), UP, buff=0.18)
        return VGroup(outer, title, cells, shape_label)

    def equation_panel(self, lines, accent=YELLOW, width=11.5, font_size=38):
        equations = VGroup(*[MathTex(line, font_size=font_size, color=WHITE_TEXT) for line in lines])
        equations.arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        panel = RoundedRectangle(
            corner_radius=0.18,
            width=width,
            height=equations.height + 0.7,
            stroke_color=accent,
            stroke_width=2,
            fill_color="#101D2E",
            fill_opacity=0.98,
        )
        equations.move_to(panel)
        return VGroup(panel, equations)

    def show_title_and_question(self):
        title = Text("What does one Transformer projection cost?", font_size=46, weight=BOLD, color=WHITE_TEXT)
        subtitle = Text("Decode · batch 1 · one new token · hidden width 4,096 · FP16 weights", font_size=23, color=MUTED)
        subtitle.next_to(title, DOWN, buff=0.28)
        formula = MathTex(r"Q = XW_Q", font_size=66, color=GREEN).next_to(subtitle, DOWN, buff=0.65)

        questions = VGroup(
            self.pill("1  Parameters", PURPLE, PURPLE_DARK),
            self.pill("2  HBM bytes", ORANGE, ORANGE_DARK),
            self.pill("3  FLOPs", GREEN, GREEN_DARK),
        ).arrange(RIGHT, buff=0.42).next_to(formula, DOWN, buff=0.7)

        boundary = Text("First understand the operation. Then count it.", font_size=24, color=YELLOW).to_edge(DOWN, buff=0.55)
        self.play(FadeIn(title, shift=UP * 0.2), FadeIn(subtitle, shift=UP * 0.2), run_time=1.2)
        self.play(Write(formula), run_time=1.0)
        self.play(LaggedStart(*[GrowFromCenter(item) for item in questions], lag_ratio=0.18), run_time=1.2)
        self.play(FadeIn(boundary), run_time=0.7)
        self.wait(1.0)
        self.play(FadeOut(VGroup(title, subtitle, formula, questions, boundary)), run_time=0.7)

    def show_shape_mechanics(self):
        header = self.header("Step 1 — See the matrix multiplication", "The matching 4,096 dimensions form each dot product")
        self.play(FadeIn(header))

        x_card = self.matrix_card("X", r"[1\times4096]", 1, 8, BLUE, BLUE_DARK, width=3.0, height=2.0).shift(LEFT * 4.25 + DOWN * 0.15)
        w_card = self.matrix_card("W_Q", r"[4096\times4096]", 7, 7, PURPLE, PURPLE_DARK, width=3.5, height=3.5).shift(DOWN * 0.15)
        q_card = self.matrix_card("Q", r"[1\times4096]", 1, 8, GREEN, GREEN_DARK, width=3.0, height=2.0).shift(RIGHT * 4.25 + DOWN * 0.15)
        multiply = MathTex(r"\times", font_size=48, color=WHITE_TEXT).move_to((x_card.get_right() + w_card.get_left()) / 2)
        equals = MathTex(r"=", font_size=48, color=WHITE_TEXT).move_to((w_card.get_right() + q_card.get_left()) / 2)

        self.play(LaggedStart(FadeIn(x_card), Write(multiply), FadeIn(w_card), Write(equals), FadeIn(q_card), lag_ratio=0.14), run_time=1.8)

        x_inner = SurroundingRectangle(x_card[3], color=YELLOW, buff=0.08)
        w_inner = SurroundingRectangle(w_card[3], color=YELLOW, buff=0.08)
        match = Text("inner dimensions match", font_size=22, color=YELLOW).to_edge(DOWN, buff=0.7)
        matching = MathTex(r"4096 = 4096", font_size=34, color=YELLOW).next_to(match, UP, buff=0.18)
        self.play(Create(x_inner), Create(w_inner), FadeIn(match), Write(matching), run_time=1.0)
        self.wait(0.8)

        # Isolate one representative row and column to reveal a dot product.
        row_highlight = SurroundingRectangle(x_card[2], color=BLUE, buff=0.06)
        column_cells = VGroup(*[w_card[2][index] for index in range(0, 49, 7)])
        col_highlight = SurroundingRectangle(column_cells, color=PURPLE, buff=0.06)
        dot_formula = MathTex(
            r"q_0 = x_0w_{0,0}+x_1w_{1,0}+\cdots+x_{4095}w_{4095,0}",
            font_size=32,
            color=WHITE_TEXT,
        ).to_edge(DOWN, buff=0.55)
        self.play(FadeOut(VGroup(match, matching, x_inner, w_inner)), Create(row_highlight), Create(col_highlight))
        self.play(ReplacementTransform(VGroup(row_highlight.copy(), col_highlight.copy()), dot_formula), run_time=1.1)
        contribution = self.pill("4,096 multiply-accumulate contributions → one output value", GREEN, GREEN_DARK, 20).next_to(dot_formula, UP, buff=0.25)
        self.play(GrowFromCenter(contribution))
        self.wait(1.0)

        all_outputs = Text("Repeat that dot-product structure for 4,096 output columns", font_size=22, color=GREEN).next_to(contribution, UP, buff=0.24)
        self.play(FadeIn(all_outputs), Indicate(q_card[2], color=GREEN, scale_factor=1.04), run_time=1.1)
        self.wait(0.8)
        self.play(FadeOut(VGroup(header, x_card, w_card, q_card, multiply, equals, row_highlight, col_highlight, dot_formula, contribution, all_outputs)), run_time=0.8)

    def show_parameter_count(self):
        header = self.header("Step 2 — Count the weights", "Every position in W_Q stores one learned parameter")
        self.play(FadeIn(header))

        grid = self.matrix_card("W_Q", r"[4096\times4096]", 8, 8, PURPLE, PURPLE_DARK, width=5.0, height=4.5).shift(LEFT * 3.5 + DOWN * 0.2)
        row_label = Text("4,096 rows", font_size=23, color=PURPLE).rotate(PI / 2).next_to(grid, LEFT, buff=0.22)
        col_label = Text("4,096 columns", font_size=23, color=PURPLE).next_to(grid, UP, buff=0.15)
        self.play(FadeIn(grid), FadeIn(row_label), FadeIn(col_label), run_time=1.0)

        panel = self.equation_panel([
            r"\text{parameters}=\text{rows}\times\text{columns}",
            r"=4096\times4096",
            r"=16{,}777{,}216",
        ], PURPLE, width=6.4).shift(RIGHT * 3.1 + DOWN * 0.15)
        self.play(FadeIn(panel[0]), Write(panel[1][0]))
        self.play(TransformFromCopy(VGroup(row_label, col_label), panel[1][1]), run_time=1.0)
        self.play(Write(panel[1][2]), Flash(panel[1][2], color=YELLOW, flash_radius=1.0), run_time=1.0)

        explanation = Text("Why not 4,096? Every input feature connects to every output feature.", font_size=21, color=YELLOW).to_edge(DOWN, buff=0.55)
        self.play(FadeIn(explanation))
        self.wait(1.1)
        self.play(FadeOut(VGroup(header, grid, row_label, col_label, panel, explanation)), run_time=0.8)

    def show_hbm_bytes(self):
        header = self.header("Step 3 — Turn parameters into HBM traffic", "Assumption: W_Q is not reusable from an on-chip cache")
        self.play(FadeIn(header))

        hbm = RoundedRectangle(width=3.2, height=4.3, corner_radius=0.18, stroke_color=ORANGE, stroke_width=3, fill_color=ORANGE_DARK, fill_opacity=0.95).shift(LEFT * 4.2 + DOWN * 0.2)
        hbm_title = Text("HBM", font_size=36, weight=BOLD, color=ORANGE).next_to(hbm.get_top(), DOWN, buff=0.25)
        weight_blocks = VGroup(*[
            RoundedRectangle(width=2.45, height=0.46, corner_radius=0.08, stroke_color=ORANGE, fill_color=ORANGE, fill_opacity=0.18)
            for _ in range(6)
        ]).arrange(DOWN, buff=0.13).move_to(hbm).shift(DOWN * 0.22)
        block_labels = VGroup(*[Text("FP16 weight values", font_size=15, color=WHITE_TEXT).move_to(block) for block in weight_blocks])
        hbm_group = VGroup(hbm, hbm_title, weight_blocks, block_labels)

        compute = RoundedRectangle(width=2.6, height=2.0, corner_radius=0.18, stroke_color=GREEN, stroke_width=3, fill_color=GREEN_DARK, fill_opacity=0.95).shift(LEFT * 1.2 + DOWN * 0.2)
        compute_title = Text("projection", font_size=25, weight=BOLD, color=GREEN).move_to(compute).shift(UP * 0.28)
        compute_sub = Text("Q = X W_Q", font_size=23, color=WHITE_TEXT).next_to(compute_title, DOWN, buff=0.24)
        compute_group = VGroup(compute, compute_title, compute_sub)
        arrows = VGroup(*[
            Arrow(weight_blocks[index].get_right(), compute.get_left() + UP * (0.65 - index * 0.26), color=ORANGE, stroke_width=3, buff=0.08)
            for index in range(6)
        ])

        self.play(FadeIn(hbm_group), FadeIn(compute_group), run_time=1.0)
        self.play(LaggedStart(*[GrowArrow(arrow) for arrow in arrows], lag_ratio=0.12), run_time=1.4)

        panel = self.equation_panel([
            r"\text{bytes}=16{,}777{,}216\ \text{weights}\times2\ \frac{\text{bytes}}{\text{weight}}",
            r"=33{,}554{,}432\ \text{bytes}",
            r"=33.55\ \text{MB}=32\ \text{MiB}",
        ], ORANGE, width=6.1, font_size=29).shift(RIGHT * 3.35 + DOWN * 0.25)
        self.play(FadeIn(panel[0]), Write(panel[1][0]), run_time=1.0)
        self.play(Write(panel[1][1]), run_time=0.7)
        self.play(Write(panel[1][2]), Flash(panel[1][2], color=YELLOW, flash_radius=1.1), run_time=1.0)

        units = VGroup(
            Text("MB uses 1,000,000 bytes", font_size=18, color=MUTED),
            Text("MiB uses 1,048,576 bytes", font_size=18, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.10).to_edge(DOWN, buff=0.42)
        self.play(FadeIn(units))
        self.wait(1.1)
        self.play(FadeOut(VGroup(header, hbm_group, compute_group, arrows, panel, units)), run_time=0.8)

    def show_flop_count(self):
        header = self.header("Step 4 — Count arithmetic", "One multiply plus one add is counted as approximately 2 FLOPs")
        self.play(FadeIn(header))

        multiply = self.pill("multiply", BLUE, BLUE_DARK, 26).shift(LEFT * 3.5 + UP * 1.5)
        plus = MathTex(r"+", font_size=52, color=WHITE_TEXT).next_to(multiply, RIGHT, buff=0.35)
        addition = self.pill("add", GREEN, GREEN_DARK, 26).next_to(plus, RIGHT, buff=0.35)
        equals = MathTex(r"=", font_size=52, color=WHITE_TEXT).next_to(addition, RIGHT, buff=0.35)
        two = self.pill("2 FLOPs", YELLOW, ORANGE_DARK, 28).next_to(equals, RIGHT, buff=0.35)
        self.play(LaggedStart(GrowFromCenter(multiply), Write(plus), GrowFromCenter(addition), Write(equals), GrowFromCenter(two), lag_ratio=0.13), run_time=1.4)

        dims = VGroup(
            self.pill("M = 1 row", BLUE, BLUE_DARK),
            self.pill("K = 4,096 contributions", PURPLE, PURPLE_DARK),
            self.pill("N = 4,096 outputs", GREEN, GREEN_DARK),
        ).arrange(RIGHT, buff=0.35).shift(UP * 0.25)
        self.play(LaggedStart(*[FadeIn(item, shift=UP * 0.15) for item in dims], lag_ratio=0.15), run_time=1.0)

        panel = self.equation_panel([
            r"\text{FLOPs}\approx2MKN",
            r"=2\times1\times4096\times4096",
            r"=33{,}554{,}432\ \text{FLOPs}",
            r"\approx33.55\ \text{MFLOPs}",
        ], GREEN, width=9.2).shift(DOWN * 2.1)
        self.play(FadeIn(panel[0]))
        for line in panel[1]:
            self.play(Write(line), run_time=0.65)
        self.play(Flash(panel[1][-1], color=YELLOW, flash_radius=1.0))
        self.wait(1.0)
        self.play(FadeOut(VGroup(header, multiply, plus, addition, equals, two, dims, panel)), run_time=0.8)

    def show_summary_and_boundary(self):
        header = self.header("One decode-time Q projection: operation-model summary")
        self.play(FadeIn(header))

        cards = VGroup(
            self.summary_card("PARAMETERS", "16,777,216", PURPLE, PURPLE_DARK),
            self.summary_card("HBM WEIGHT READ", "33.55 MB  /  32 MiB", ORANGE, ORANGE_DARK),
            self.summary_card("COMPUTE", "33.55 MFLOPs", GREEN, GREEN_DARK),
        ).arrange(RIGHT, buff=0.35).shift(UP * 1.35)
        self.play(LaggedStart(*[GrowFromCenter(card) for card in cards], lag_ratio=0.16), run_time=1.4)

        observation = RoundedRectangle(width=11.9, height=1.25, corner_radius=0.16, stroke_color=YELLOW, stroke_width=2, fill_color="#222B30", fill_opacity=1).shift(DOWN * 0.15)
        obs_title = Text("Why are bytes and FLOPs numerically similar?", font_size=24, weight=BOLD, color=YELLOW).next_to(observation.get_top(), DOWN, buff=0.18)
        obs_text = Text("M=1: each FP16 weight is 2 bytes and causes about 2 FLOPs. Same number—different units.", font_size=20, color=WHITE_TEXT).next_to(obs_title, DOWN, buff=0.18)
        self.play(FadeIn(observation), FadeIn(obs_title), FadeIn(obs_text))

        included = VGroup(
            Text("INCLUDED", font_size=18, weight=BOLD, color=GREEN),
            Text("W_Q HBM read · projection multiply-adds", font_size=18, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        excluded = VGroup(
            Text("EXCLUDED", font_size=18, weight=BOLD, color=RED),
            Text("X read · Q write · bias · launch · K/V · attention · cache reuse", font_size=18, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        boundary = VGroup(included, excluded).arrange(DOWN, aligned_edge=LEFT, buff=0.35).shift(DOWN * 2.0 + LEFT * 1.0)
        self.play(FadeIn(boundary, shift=UP * 0.15))

        closing = Text("A useful performance model always states its boundary.", font_size=24, color=YELLOW).to_edge(DOWN, buff=0.34)
        self.play(FadeIn(closing))
        self.wait(2.0)

    def summary_card(self, title, value, color, fill):
        box = RoundedRectangle(width=3.85, height=1.55, corner_radius=0.16, stroke_color=color, stroke_width=3, fill_color=fill, fill_opacity=0.96)
        heading = Text(title, font_size=17, weight=BOLD, color=color).next_to(box.get_top(), DOWN, buff=0.22)
        result = Text(value, font_size=27, weight=BOLD, color=WHITE_TEXT).next_to(heading, DOWN, buff=0.25)
        return VGroup(box, heading, result)
