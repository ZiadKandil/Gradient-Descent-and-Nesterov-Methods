#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <cmath>
#include <fstream>
#include <sstream>
#include "mpParser.h"


using Vector = std::vector<double>;

/**
 * @enum Update rules for learning rate and gradient computation
 */
enum class AlphaRule{
    EXP_DECAY,
    INV_DECAY,
    LINE_SEARCH
};

/**
 * @enum Update rules for gradient computation
 */
enum class GradRule{
    ANALYTIC,
    FINITE_DIFF
};

// Struct to hold the parameters for the optimization problem
struct Params {
    std::function<double(const Vector &)> f;  /// < Main function to minimize
    std::function<Vector(const Vector &)> grad_f;  /// < Gradient of the main function
    Vector x0;  /// < Initial guess
    double step_tol;  /// < Tolerance for step size
    double res_tol;  /// < Tolerance for residual
    double alpha0;  /// < Initial learning rate
    int max_iters;  /// < Maximum number of iterations
    double sigma;  /// < Line search parameter
    AlphaRule alpha_rule;  /// < Learning rate update rule
    GradRule grad_rule;  /// < Gradient computation rule
};

/**
 * @brief Function to compute the norm of a vector
 * @param v The input vector
 * @return The Euclidean norm of the vector
 */
double norm(const Vector &v){
    double sum = 0.0;
    for (double val : v) {
        sum += val * val;
    }
    return std::sqrt(sum);
};

/**
 * @brief Function to compute the square of the norm of a vector for efficiency
 * @param v The input vector
 * @return The square of the Euclidean norm of the vector
 */
double norm_squared(const Vector &v){
    double sum = 0.0;
    for (double val : v) {
        sum += val * val;
    }
    return sum;
};

/**
 * @brief Operator overload for vector subtraction
 * @param a The first vector
 * @param b The second vector
 * @return The result of subtracting vector b from vector a
 * @throws std::invalid_argument if the vectors are of different sizes
 */
Vector operator-(const Vector &a, const Vector &b){
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same size for subtraction");
    }
    Vector result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
};

/**
 * @brief Operator overload to handle vector multiplication by a scalar
 * @param scalar The scalar value
 * @param v The input vector
 * @return The result of multiplying the vector by the scalar
 */
Vector operator*(const double scalar, const Vector &v){
    Vector result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = v[i] * scalar;
    }
    return result;
};

/**
 * @brief Operator overload for vector addition
 * @param a The first vector
 * @param b The second vector
 * @return The result of adding vector a and vector b
 * @throws std::invalid_argument if the vectors are of different sizes
 */
Vector operator+(const Vector &a, const Vector &b){
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same size for subtraction");
    }
    Vector result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
};

/**
 * @brief Function to update learning rate using exponential decay
 * @param alpha0 The initial learning rate
 * @param iter The current iteration number
 * @param decay_rate The decay rate (default is 0.2)
 * @return The updated learning rate after applying exponential decay
 */
double exp_decay(double alpha0, int iter, double decay_rate=0.2) {
    return alpha0 * std::exp(-decay_rate * iter);
};

/**
 * @brief Function to update learning rate using inverse decay
 * @param alpha0 The initial learning rate
 * @param iter The current iteration number
 * @param decay_rate The decay rate (default is 0.2)
 * @return The updated learning rate after applying inverse decay
 */
double inv_decay(double alpha0, int iter, double decay_rate=0.2) {
    return alpha0 / (1 + decay_rate * iter);
};

/**
 * @brief Function to perform line search using Armijo's rule
 * @param f The main function
 * @param grad The gradient of the main function
 * @param x The current point
 * @param alpha0 The initial step size
 * @param sigma The line search parameter
 * @return The computed step size
 */
double line_search(
    const std::function<double(const Vector &)> &f,
    const Vector &grad,
    const Vector &x,
    const double alpha0,
    const double sigma
){
    double alpha = alpha0;
    double f_x = f(x);
    double grad_norm_squared = norm_squared(grad);
    while(sigma * alpha * grad_norm_squared > f_x - f(x - alpha * grad)){
        alpha *= 0.5;  // Reduce alpha by half
    }
    return alpha;
};

/**
 * @brief Function to compute the gradient using finite difference approximation
 * @param f The main function
 * @param x The point at which to compute the gradient
 * @param h The perturbation size for finite difference (default is 1e-6)
 * @return The computed gradient vector
 */
Vector finite_diff_grad(
    const std::function<double(const Vector &)> &f,
    const Vector &x,
    double h = 1e-6
){
    Vector grad(x.size());
    Vector x_forward = x;
    Vector x_backward = x;

    for(size_t i = 0; i < x.size(); ++i){
        x_forward[i] += h;  // Perturb the i-th component forward
        x_backward[i] -= h; // Perturb the i-th component backward
        grad[i] = (f(x_forward) - f(x_backward)) / (2 * h);  // Central difference formula
        x_forward[i] = x[i];
        x_backward[i] = x[i];
    }
    return grad;
};

/**
 * @brief Function to parse alpha rule from string (configuration file)
 * @param rule_str The string representing the alpha rule
 * @return The parsed alpha rule
 * @throws std::runtime_error if the rule is unknown
 */
AlphaRule parseAlphaRule(const std::string &rule_str) {
    if (rule_str == "EXP_DECAY") return AlphaRule::EXP_DECAY;
    if (rule_str == "INV_DECAY") return AlphaRule::INV_DECAY;
    if (rule_str == "LINE_SEARCH") return AlphaRule::LINE_SEARCH;
    throw std::runtime_error("Unknown alpha rule: " + rule_str);
}

/**
 * @brief Function to parse gradient rule from string (configuration file)
 * @param rule_str The string representing the gradient rule
 * @return The parsed gradient rule
 * @throws std::runtime_error if the rule is unknown
 */
GradRule parseGradRule(const std::string &rule_str) {
    if (rule_str == "ANALYTIC") return GradRule::ANALYTIC;
    if (rule_str == "FINITE_DIFF") return GradRule::FINITE_DIFF;
    throw std::runtime_error("Unknown gradient rule: " + rule_str);
}

/**
 * @brief Function to read configuration from a file and populate the parameters struct
 * @param filename The name of the configuration file
 * @param P The parameters struct to populate with the configuration values
 * @throws std::runtime_error if the file cannot be opened or if the configuration is invalid
 */
void read_config(const std::string &filename, Params &P){

    std::ifstream file(filename);

    // Check if the file was opened successfully
    if(!file.is_open()){
        throw std::runtime_error("Could not open configuration file: " + filename);
    }

    std::string line;
    std::vector<std::string> lines;

    // Read the file line by line

    while(std::getline(file, line)){
        // Skip empty lines and comments
        if(line.empty() || line[0] == '#'){
            continue;
        }

        // Trim whitespace from the line
        line.erase(0, line.find_first_not_of(" \t\n\r"));  // Remove white space from the begining to the first non-whitespace character
        line.erase(line.find_last_not_of(" \t\n\r") + 1);  // Remove white space from the end to the last non-whitespace character

        // If line is not empty after trimming then store for further processing
        if(!line.empty()){
            lines.push_back(line);
        }
    }

    file.close();

    if(lines.size() < 7){
        throw std::runtime_error("Configuration file must contain at least 7 lines \
              for function, gradient, initial guess, tolerances, learning rate, \
              max iterations, and line search parameter.");
    }

    // Line 0: Main function
    std::string f_expr = lines[0];

    // Line 1: Gradient (comma separated expressions for each component)
    std::string grad_expr = lines[1];

    // Line 2: Initial guess (comma separated values)
    std::istringstream iss(lines[2]);
    double x0_0, x0_1;
    iss >> x0_0 >> x0_1;
    P.x0 = {x0_0, x0_1};

    // Line 3: Step size tolerance
    P.step_tol = std::stod(lines[3]);  // stod = string to double

    // Line 4: Residual tolerance
    P.res_tol = std::stod(lines[4]);

    // Line 5: Initial learning rate
    P.alpha0 = std::stod(lines[5]);

    // Line 6: Maximum number of iterations
    P.max_iters = std::stoi(lines[6]);  // stoi = string to int

    // Line 7: sigma (if exists)
    if (lines.size() > 7) {
        P.sigma = std::stod(lines[7]);
    } else {
        P.sigma = 0.5;
    };

    // Line 8: Alpha rule (if exists)
    if (lines.size() > 8) {
        P.alpha_rule = parseAlphaRule(lines[8]);
    } else {
        P.alpha_rule = AlphaRule::LINE_SEARCH;
    }

    // Line 9: Gradient rule (if exists)
    if (lines.size() > 9) {
        P.grad_rule = parseGradRule(lines[9]);
    } else {
        P.grad_rule = GradRule::FINITE_DIFF;
    }

    // Create function using muparserx
    P.f = [f_expr](const Vector &x){
        mup::ParserX parser;
        mup::Value xa(x[0]), xb(x[1]);
        parser.DefineVar("x1", mup::Variable(&xa));
        parser.DefineVar("x2", mup::Variable(&xb));
        parser.SetExpr(f_expr);
        return parser.Eval().GetFloat();
    };

    // Create gradient function using muparserx
    P.grad_f = [grad_expr](const Vector &x){
        mup::ParserX parser;
        mup::Value xa(x[0]), xb(x[1]);
        parser.DefineVar("x1", mup::Variable(&xa));
        parser.DefineVar("x2", mup::Variable(&xb));
        parser.SetExpr(grad_expr);
        
        mup::Value result = parser.Eval();
        Vector grad(2);
        grad[0] = result.At(0).GetFloat();  // First component of the gradient
        grad[1] = result.At(1).GetFloat();  // Second component of the gradient
        return grad;
    };
}



/**
 * @brief Template function to perform gradient method using different learning rate and gradient update rules
 * @tparam alpha_rule The alpha rule to use
 * @tparam grad_rule The gradient rule to use
 * @param P The parameters struct
 * @return The value of the function at the minimum point
 */
template <AlphaRule alpha_rule, GradRule grad_rule>
double Gradient(const Params &P){

    std::cout << "Running Gradient method with " 
              << (alpha_rule == AlphaRule::EXP_DECAY ? "Exponential Decay" : (alpha_rule == AlphaRule::INV_DECAY ? "Inverse Decay" : "Line Search"))
              << " learning rate and "
              << (grad_rule == GradRule::ANALYTIC ? "Analytic" : "Finite Difference")
              << " gradient." << std::endl;

    Vector x = P.x0;  // Initial guess
    double alpha = P.alpha0;  // Initial learning rate
    Vector x_old = x;  // To store the previous point for step size convergence check
    Vector grad(x.size());  // To store the gradient

    for (int iter = 0; iter < P.max_iters; ++iter){

        if constexpr (grad_rule == GradRule::ANALYTIC) {
            grad = P.grad_f(x);  // Use the provided analytic gradient
        } else if constexpr (grad_rule == GradRule::FINITE_DIFF) {
            grad = finite_diff_grad(P.f, x);  // Compute the gradient using finite difference
        }

        double grad_norm = norm(grad);  // Compute the norm of the gradient

        // Check for convergence based on the norm of the gradient
        if (grad_norm < P.res_tol) {
            std::cout << "Residual convergence achieved using gradient method at iteration " << iter << std::endl;
            return P.f(x);  // Return the value of the function at the minimum point
        }

        // Update learning rate based on the specified rule
        if constexpr (alpha_rule == AlphaRule::EXP_DECAY) {
            alpha = exp_decay(P.alpha0, iter);
        } else if constexpr (alpha_rule == AlphaRule::INV_DECAY) {
            alpha = inv_decay(P.alpha0, iter);
        } else if constexpr (alpha_rule == AlphaRule::LINE_SEARCH) {
            alpha = line_search(P.f, grad, x, alpha, P.sigma);
        }

        // Update the current point
        x = x - alpha * grad;

        // Check for convergence based on the step size
        if (norm(x - x_old) < P.step_tol) {
            std::cout << "Step size convergence achieved using gradient method at iteration " << iter << std::endl;
            return P.f(x);  // Return the value of the function at the minimum point
        }
        x_old = x;  // Update the old point for the next iteration
    }

    std::cout << "Maximum iterations reached using gradient method without convergence." << std::endl;
    return P.f(x);  // Return the value of the function at the last point
}

/**
 * @brief Template function to perform Nesterov'st method using different learning rate and gradient update rules
 * @tparam alpha_rule The alpha rule to use
 * @tparam grad_rule The gradient rule to use
 * @param P The parameters struct
 * @return The value of the function at the minimum point
 */
template <AlphaRule alpha_rule, GradRule grad_rule>
double Nesterov(const Params &P){

    std::cout << "Running Nesterov method with " 
              << (alpha_rule == AlphaRule::EXP_DECAY ? "Exponential Decay" : (alpha_rule == AlphaRule::INV_DECAY ? "Inverse Decay" : "Line Search"))
              << " learning rate and "
              << (grad_rule == GradRule::ANALYTIC ? "Analytic" : "Finite Difference")
              << " gradient." << std::endl;

    Vector y = P.x0;  // Initial guess
    Vector x = y;  // To store the current point
    double alpha = P.alpha0;  // Initial learning rate
    Vector y_old = y;  // To store the previous point for step size convergence check
    double eta;  // Memory parameter
    double t = 1.0;
    double t_old = t;
    Vector grad(y.size());  // To store the gradient
    Vector x_old = x;  // To store the previous point for step size convergence check

    for(int k = 0; k < P.max_iters; ++k){

        // Compute the gradient at the current point based on the specified gradient rule
        if constexpr (grad_rule == GradRule::ANALYTIC) {
            grad = P.grad_f(y);  // Use the provided analytic gradient
        } else if constexpr (grad_rule == GradRule::FINITE_DIFF) {
            grad = finite_diff_grad(P.f, y);  // Compute the gradient using finite difference
        }

        double grad_norm = norm(grad);  // Compute the norm of the gradient

        // Check for convergence based on the norm of the gradient
        if (grad_norm < P.res_tol) {
            std::cout << "Residual convergence achieved using Nesterov method at iteration " << k << std::endl;
            return P.f(x);  // Return the value of the function at the minimum point
        }

        // Update learning rate based on the specified rule
        if constexpr (alpha_rule == AlphaRule::EXP_DECAY) {
            alpha = exp_decay(P.alpha0, k);
        } else if constexpr (alpha_rule == AlphaRule::INV_DECAY) {
            alpha = inv_decay(P.alpha0, k);
        } else if constexpr (alpha_rule == AlphaRule::LINE_SEARCH) {
            alpha = line_search(P.f, grad, y, alpha, P.sigma);
        }

        x = y - alpha * grad;  // Update the current point

        // Check for convergence based on the step size
        if (norm(x - x_old) < P.step_tol) {
            std::cout << "Step size convergence achieved using Nesterov method at iteration " << k << std::endl;
            return P.f(x);  // Return the value of the function at the minimum point
        }

        t = (1 + std::sqrt(1 + t_old * t_old)) / 2;  // Update memory parameter
        eta = (t_old - 1) / t;  // Compute momentum term

        y_old = y;  // Store old point for step size convergence check
        y = x + eta * (x - x_old);  // Update y with momentum

        x_old = x;  // Update old point for next iteration
        t_old = t;  // Update old memory parameter for next iteration
    }

    std::cout << "Maximum iterations reached using Nesterov method without convergence." << std::endl;
    return P.f(x);  // Return the value of the function at the last point
}

/**
 * @brief Function to call the appropriate gradient method based on update rules specified in the parameters
 * @param P The parameters struct
 * @return The value of the function at the minimum point
 */
double runGradient(const Params &P) {
    if (P.alpha_rule == AlphaRule::EXP_DECAY && P.grad_rule == GradRule::ANALYTIC)
        return Gradient<AlphaRule::EXP_DECAY, GradRule::ANALYTIC>(P);
    if (P.alpha_rule == AlphaRule::EXP_DECAY && P.grad_rule == GradRule::FINITE_DIFF)
        return Gradient<AlphaRule::EXP_DECAY, GradRule::FINITE_DIFF>(P);
    if (P.alpha_rule == AlphaRule::INV_DECAY && P.grad_rule == GradRule::ANALYTIC)
        return Gradient<AlphaRule::INV_DECAY, GradRule::ANALYTIC>(P);
    if (P.alpha_rule == AlphaRule::INV_DECAY && P.grad_rule == GradRule::FINITE_DIFF)
        return Gradient<AlphaRule::INV_DECAY, GradRule::FINITE_DIFF>(P);
    if (P.alpha_rule == AlphaRule::LINE_SEARCH && P.grad_rule == GradRule::ANALYTIC)
        return Gradient<AlphaRule::LINE_SEARCH, GradRule::ANALYTIC>(P);
    if (P.alpha_rule == AlphaRule::LINE_SEARCH && P.grad_rule == GradRule::FINITE_DIFF)
        return Gradient<AlphaRule::LINE_SEARCH, GradRule::FINITE_DIFF>(P);
    throw std::runtime_error("Invalid rule combination");
}

/**
 * @brief Function to call the appropriate Nesterov method based on update rules specified in the parameters
 * @param P The parameters struct
 * @return The value of the function at the minimum point
 */
double runNesterov(const Params &P) {
    if (P.alpha_rule == AlphaRule::EXP_DECAY && P.grad_rule == GradRule::ANALYTIC)
        return Nesterov<AlphaRule::EXP_DECAY, GradRule::ANALYTIC>(P);
    if (P.alpha_rule == AlphaRule::EXP_DECAY && P.grad_rule == GradRule::FINITE_DIFF)
        return Nesterov<AlphaRule::EXP_DECAY, GradRule::FINITE_DIFF>(P);
    if (P.alpha_rule == AlphaRule::INV_DECAY && P.grad_rule == GradRule::ANALYTIC)
        return Nesterov<AlphaRule::INV_DECAY, GradRule::ANALYTIC>(P);
    if (P.alpha_rule == AlphaRule::INV_DECAY && P.grad_rule == GradRule::FINITE_DIFF)
        return Nesterov<AlphaRule::INV_DECAY, GradRule::FINITE_DIFF>(P);
    if (P.alpha_rule == AlphaRule::LINE_SEARCH && P.grad_rule == GradRule::ANALYTIC)
        return Nesterov<AlphaRule::LINE_SEARCH, GradRule::ANALYTIC>(P);
    if (P.alpha_rule == AlphaRule::LINE_SEARCH && P.grad_rule == GradRule::FINITE_DIFF)
        return Nesterov<AlphaRule::LINE_SEARCH, GradRule::FINITE_DIFF>(P);
    throw std::runtime_error("Invalid rule combination");
}

int main(int argc, char *argv[]){

    Params P;

    // If a configuration file is provided as a command-line argument, read the parameters from the file
    if(argc > 1){
        try{
            read_config(argv[1], P);
            std::cout << "Configuration loaded successfully from " << argv[1] << std::endl;
        }
        catch (const std::exception &e){
            std::cerr << "Error reading configuration: " << e.what() << std::endl;
            return 1;
        }
    } else {
        std::cout << "No configuration file provided. Using hardcoded parameters." << std::endl;

        // Define the main function and its gradient
        P.f = [](const Vector &x) {
            return x[0] * x[1] + 4 * std::pow(x[0], 4) + x[1] * x[1] + 3 * x[0];  
        };

        P.grad_f = [](const Vector &x) {
            return Vector{x[1] + 16 * std::pow(x[0], 3) + 3, x[0] + 2 * x[1]};
        };

        P.x0 = {0.0, 0.0};  // Initial guess
        P.step_tol = 1e-6;  // Tolerance for step size
        P.res_tol = 1e-6;  // Tolerance for residual
        P.alpha0 = 0.01;  // Initial learning rate
        P.max_iters = 100000;  // Maximum number of iterations
        P.sigma = 0.5;  // Line search parameter
        P.alpha_rule = AlphaRule::LINE_SEARCH;  // Learning rate update rule
        P.grad_rule = GradRule::FINITE_DIFF;  // Gradient computation rule

    }

    double result_grad = runGradient(P);
    std::cout << "Minimum value using Gradient method is: " << result_grad << std::endl;

    double result_nestrov = runNesterov(P);
    std::cout << "Minimum value using Nesterov method is: " << result_nestrov << std::endl;

    return 0;
}