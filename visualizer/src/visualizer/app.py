"""
OMDS Regression Data Analysis - BeeWare Briefcase App

This app demonstrates how to use the regplotter function from omds_functions.py
with the Briefcase framework.
"""

import toga
from toga.style.pack import COLUMN, Pack

import pandas as pd
import numpy as np

from .omds_functions import regplotter


class RegressionPlotter(toga.App):
    def startup(self):
        """Construct and show the Toga application."""
        
        # Create main container
        main_box = toga.Box(style=Pack(
            direction=COLUMN,
            padding=10,
            flex=1,
        ))
        
        # Add title
        title = toga.Label(
            'Regression Plotter',
            style=Pack(
                padding=5,
                font_size=18,
                font_weight='bold',
            )
        )
        main_box.add(title)
        
        # Create a section for instructions
        instructions = toga.Label(
            'This app demonstrates regression analysis visualization.\n'
            'Click the button below to generate a sample regression plot.',
            style=Pack(
                padding=10,
                font_size=11,
            )
        )
        main_box.add(instructions)
        
        # Create button to generate plot
        generate_button = toga.Button(
            'Generate Sample Plot',
            on_press=self.action_generate_plot,
            style=Pack(
                padding=10,
                width=200,
            )
        )
        main_box.add(generate_button)
        
        # Create output text area
        self.output_text = toga.MultilineTextInput(
            readonly=True,
            style=Pack(
                flex=1,
                padding=10,
                width=400,
                height=200,
            )
        )
        main_box.add(self.output_text)
        
        # Create main window
        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = main_box
        self.main_window.show()
    
    def action_generate_plot(self, widget):
        """Generate a sample regression plot."""
        try:
            # Create sample data
            np.random.seed(42)
            n_samples = 100
            
            # Generate synthetic data
            x = np.linspace(0, 100, n_samples)
            y = 2.5 * x + np.random.normal(0, 50, n_samples)
            group = np.random.choice(['Group A', 'Group B', 'Group C'], n_samples)
            
            df = pd.DataFrame({
                'feature1': x,
                'feature2': y,
                'group': group,
            })
            
            # Generate plot (saved to file)
            slope, intercept, r_squared, p_value = regplotter(
                df,
                'feature1', 'Feature 1 (X-axis)',
                'feature2', 'Feature 2 (Y-axis)',
                'group', 'Group Classification',
                output_mode='file'
            )
            
            # Update output text
            output = f"""
Plot Generated Successfully!

Regression Statistics:
  Slope: {slope:.4f}
  Intercept: {intercept:.4f}
  R²: {r_squared:.4f}
  P-value: {p_value}

Interpretation:
  - The relationship shows a slope of {slope:.2f}
  - R² of {r_squared:.4f} indicates {'strong' if r_squared > 0.5 else 'moderate' if r_squared > 0.2 else 'weak'} correlation
  - Plot has been saved to the app data directory

To use your own data:
  1. Load your CSV file to a pandas DataFrame
  2. Call regplotter() with your data
  3. Use output_mode='file' to save, or 'bytes' for image data
            """
            
            self.output_text.value = output
            
        except Exception as e:
            self.output_text.value = f"Error generating plot:\n{str(e)}"


def main():
    return RegressionPlotter(
        'Regression Plotter',
        'com.example.visualizer'
    )
