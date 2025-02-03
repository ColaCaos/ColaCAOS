from manim import *

class LogisticCobwebZoom(Scene):
    def construct(self):
        # PARAMETERS
        r = 1.8
        x_start = 0.2  # Must be in [0,1] for this demonstration
        n_iterations = 20  # Fewer iterations just for clarity in the small region
        
        # Define logistic function: f(x) = r*x*(1 - x)
        def logistic(x):
            return r * x * (1 - x)
        
        # Create axes that only show x in [0,1], y in [0,1].
        axes = Axes(
            x_range=[0, 1, 0.1],
            y_range=[0, 1, 0.1],
            x_length=6,
            y_length=6,
            tips=False,  # turn off arrow tips if desired
            axis_config={
                "include_numbers": True,
                "font_size": 24,
            }
        )
        axes.to_edge(DOWN)  # shift the axes a bit; optional

        # Labels
        x_label = axes.get_x_axis_label("x", edge=RIGHT, direction=UR)
        y_label = axes.get_y_axis_label("f(x)", edge=UP, direction=UR)

        # Plot logistic curve on [0,1]
        logistic_graph = axes.plot(
            logistic,
            x_range=[0, 1],  # Restrict plot to [0,1]
            color=BLUE
        )
        # Line y = x on [0,1]
        line_y_equals_x = axes.plot(
            lambda x: x,
            x_range=[0, 1],
            color=GREEN
        )

        # Add axes and curves
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(logistic_graph), Create(line_y_equals_x))
        
        # Perform cobweb iteration
        x_value = x_start
        for i in range(n_iterations):
            new_x = logistic(x_value)
            
            # If you'd like, you can skip drawing if values go out of [0,1].
            # But with r=1 and x in [0,1], it should stay in [0,1].
            # If you want to guard:
            # if not (0 <= x_value <= 1 and 0 <= new_x <= 1):
            #     break

            # Vertical line from (x, x) to (x, f(x))
            vertical_line = Line(
                axes.c2p(x_value, x_value),
                axes.c2p(x_value, new_x),
                color=YELLOW
            )
            # Horizontal line from (x, f(x)) to (f(x), f(x))
            horizontal_line = Line(
                axes.c2p(x_value, new_x),
                axes.c2p(new_x, new_x),
                color=YELLOW
            )

            run_time = 0.7 if i < 5 else 0.3
            self.play(Create(vertical_line), run_time=run_time)
            self.play(Create(horizontal_line), run_time=run_time)

            x_value = new_x
        
        self.wait()
