from manim import *

class ShowLogisticMap(Scene):
    def construct(self):
        #
        # 1) Display the equation x_{n+1} = r x_n (1 - x_n) and r = 3
        #
        r_value = 3.7
        equation = MathTex(r"x_{n+1} = r\,x_n\,(1 - x_n)", font_size=36)
        equation.to_corner(UR)
        r_label = MathTex(fr"r = {r_value}", font_size=32).next_to(equation, DOWN, buff=0.3)
        self.play(Write(equation), Write(r_label))

        #
        # 2) Left plot: x_{n} on the horizontal axis, x_{n+1} on the vertical axis.
        #
        left_axes = Axes(
            x_range=[0, 1, 0.1],
            y_range=[0, 1, 0.1],
            x_length=5,
            y_length=5,
            tips=False,
            axis_config={
                "include_ticks": True,
                "include_numbers": False,  # we'll add numeric labels manually
            },
        )
        left_axes.to_edge(LEFT)

        # Numeric labels only at 0 and 1 for each axis
        left_axes.x_axis.add_numbers([0, 1], font_size=24)
        left_axes.y_axis.add_numbers([0, 1], font_size=24)

        # Axis labels: x_n (horizontal), x_{n+1} (vertical)
        left_labels = left_axes.get_axis_labels(
            MathTex(r"x_n"), 
            MathTex(r"x_{n+1}")
        )

        # Plot the curve x_{n+1} = 3 * x_n * (1 - x_n)
        logistic_curve = left_axes.plot(
            lambda x: r_value * x * (1 - x),
            x_range=[0, 1],
            color=BLUE
        )
        # Plot line x_{n+1} = x_n
        line_y_eq_x = left_axes.plot(
            lambda x: x,
            x_range=[0, 1],
            color=GREEN
        )

        # Animate creation of left plot
        self.play(Create(left_axes), Write(left_labels))
        self.play(Create(logistic_curve), Create(line_y_eq_x))

        #
        # 3) Right plot: discrete time series n vs. x_n
        #
        right_axes = Axes(
            x_range=[0, 20, 1],  # integer ticks up to 20
            y_range=[0, 1, 0.1],
            x_length=5,
            y_length=5,
            tips=False,
            axis_config={
                "include_ticks": True,
                "include_numbers": False,
            },
        )
        right_axes.to_edge(RIGHT)

        # Numeric labels for n=0,1,5,10,15,20
        right_axes.x_axis.add_numbers([0, 1, 5, 10, 15, 20], font_size=24)
        # Numeric labels for x_n = 0,1
        right_axes.y_axis.add_numbers([0, 1], font_size=24)

        # Optional axis labels: n (horizontal), x_n (vertical)
        right_labels = right_axes.get_axis_labels("n", MathTex(r"x_n"))

        # Animate creation of right plot
        self.play(Create(right_axes), Write(right_labels))

        #
        # 4) Perform the iterations & animate
        #
        x_value = 0.2       # initial seed in [0,1]
        n_iterations = 15   # do 15 iterations

        def logistic(x):
            return r_value * x * (1 - x)

        for i in range(n_iterations):
            new_x = logistic(x_value)

            # -- COBWEB LINES on the left plot --
            vertical_line = Line(
                left_axes.c2p(x_value, x_value),
                left_axes.c2p(x_value, new_x),
                color=YELLOW
            )
            horizontal_line = Line(
                left_axes.c2p(x_value, new_x),
                left_axes.c2p(new_x, new_x),
                color=YELLOW
            )

            # Slow for first 5 iterations, then faster
            run_time = 1.0 if i < 5 else 0.3
            self.play(Create(vertical_line), run_time=run_time)
            self.play(Create(horizontal_line), run_time=run_time)

            # -- DOT on the right plot (n vs. x_n) --
            dot = Dot(
                right_axes.c2p(i + 1, new_x),
                radius=0.06,
                color=RED
            )
            self.play(FadeIn(dot), run_time=run_time)

            x_value = new_x

        self.wait(2)
