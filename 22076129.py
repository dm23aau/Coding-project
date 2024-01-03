# Import Statements
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def load_and_describe_data(file_path):
    # Load data from CSV file and provide summary statistics
    data = pd.read_csv(file_path, header=None, names=['Income'])
    incomes = data['Income']
    print(data.describe())
    return incomes

def fit_normal_distribution(data):
    # Fit a normal distribution to the data and return the parameters
    fit_params = norm.fit(data)
    return fit_params

def plot_histogram_and_pdf(data, fit_params):
    # Plot histogram of incomes
    plt.hist(data, bins=30, density=True, alpha=0.6, color='skyblue', label='Income Histogram')
    
    # Plot the PDF curve using the fitted parameters
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 1000000)
    pdf_values = norm.pdf(x, *fit_params)
    plt.plot(x, pdf_values, 'r', linewidth=3, label='PDF of Incomes')

def customize_plot_labels_and_legend():
    # Customize labels and title of the plot
    plt.xlabel('Income (Euros)')
    plt.ylabel('Probability Density')
    plt.title('Income Distribution')
    
    # legend position 
    plt.legend(loc='upper right')

def calculate_mean_and_X(data, fit_params):
    # Calculate mean using trapezoidal rule for numerical integration
    x = np.linspace(min(data), max(data), 1000000)
    p = norm.pdf(x, *fit_params)
    
    mean_income = np.trapz(x * p, x)
    
    # Calculate X from the PDF
    X = norm.cdf(1.2 * mean_income, *fit_params) - norm.cdf(0.8 * mean_income, *fit_params)
    
    return mean_income, X

def annotate_mean_and_X(mean_income, X):
    # Annotate mean and X on the graph at specified coordinates
    plt.annotate(f'Mean: {mean_income:.2f} Euros', xy=(0.3, 0.6), xycoords='axes fraction')
    plt.annotate(f'X: {X:.2f}', xy=(0.3, 0.45), xycoords='axes fraction')

def show_plot():
    # Display the plot
    plt.show()

# Main execution
file_path = 'data9.csv'
incomes = load_and_describe_data(file_path)
fit_params = fit_normal_distribution(incomes)

plot_histogram_and_pdf(incomes, fit_params)
customize_plot_labels_and_legend()

mean_income, X = calculate_mean_and_X(incomes, fit_params)
annotate_mean_and_X(mean_income, X)

show_plot()
