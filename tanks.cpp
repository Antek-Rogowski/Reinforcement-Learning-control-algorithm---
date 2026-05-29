#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>

namespace py = pybind11;

class ThreeTankSystem {
private:
    // Parametry fizyczne obiektu
    const double A = 0.0154;      // Pole przekroju zbiornikow [m^2]
    const double a1 = 0.00005;    // Przekroj zaworu 1-2
    const double a2 = 0.00005;    // Przekroj zaworu 2-3
    const double a3 = 0.00005;    // Przekroj zaworu odplywowego
    const double g = 9.81;        // Przyspieszenie ziemskie
    const double max_h = 1.0;     // Maksymalna wysokosc zbiornika [m]

    // Stan ukladu: poziomy w zbiornikach [h1, h2, h3]
    std::vector<double> state;
    
    // Krok calkowania solvera [s]
    double dt;

    // Pochodne stanow (prawo Torricellego)
    std::vector<double> derivatives(const std::vector<double>& h, double q_in) {
        auto flow = [&](double h_out, double h_in, double a) {
            double delta = h_out - h_in;
            return a * std::copysign(1.0, delta) * std::sqrt(2 * g * std::abs(delta));
        };

        double q12 = flow(h[0], h[1], a1);
        double q23 = flow(h[1], h[2], a2);
        double q3_out = flow(h[2], 0.0, a3);

        return {
            (q_in - q12) / A,
            (q12 - q23) / A,
            (q23 - q3_out) / A
        };
    }

public:
    ThreeTankSystem(double delta_t = 0.01) : dt(delta_t) {
        reset();
    }

    void reset() {
        state = {0.0, 0.0, 0.0};
    }

    std::vector<double> get_state() const {
        return state;
    }

    // Wykonanie N krokow calkowania RK4 dla jednej akcji agenta
    std::vector<double> step(double action_q_in, int steps_per_action = 10) {
        // ZWOLNIENIE GIL
        py::gil_scoped_release release;

        for (int i = 0; i < steps_per_action; ++i) {
            std::vector<double> k1 = derivatives(state, action_q_in);
            
            std::vector<double> h_k2 = {state[0] + 0.5*dt*k1[0], state[1] + 0.5*dt*k1[1], state[2] + 0.5*dt*k1[2]};
            std::vector<double> k2 = derivatives(h_k2, action_q_in);
            
            std::vector<double> h_k3 = {state[0] + 0.5*dt*k2[0], state[1] + 0.5*dt*k2[1], state[2] + 0.5*dt*k2[2]};
            std::vector<double> k3 = derivatives(h_k3, action_q_in);
            
            std::vector<double> h_k4 = {state[0] + dt*k3[0], state[1] + dt*k3[1], state[2] + dt*k3[2]};
            std::vector<double> k4 = derivatives(h_k4, action_q_in);

            for (int j = 0; j < 3; ++j) {
                state[j] += (dt / 6.0) * (k1[j] + 2*k2[j] + 2*k3[j] + k4[j]);
                
                // Zabezpieczenie przed ujemnym poziomem wody
                if (state[j] < 0.0) state[j] = 0.0;
                
                // Zabezpieczenie przed przelaniem - fizyczny limit poziomu w zbiorniku
                if (state[j] > max_h) state[j] = max_h;
            }
        }

        return state;
    }
};

// --- BINDING PYBIND11 ---
PYBIND11_MODULE(tank_sim, m) {
    m.doc() = "Wydajny symulator ukladu trzech zbiornikow C++ w locie";

    py::class_<ThreeTankSystem>(m, "ThreeTankSystem")
        .def(py::init<double>(), py::arg("delta_t") = 0.01)
        .def("reset", &ThreeTankSystem::reset, "Resetuje stany obiektu do zera")
        .def("get_state", &ThreeTankSystem::get_state, "Pobiera aktualny wektor stanow (poziomy)")
        .def("step", &ThreeTankSystem::step, "Wykonuje kroki symulacji RK4 i zwraca nowe stany",
             py::arg("action_q_in"), py::arg("steps_per_action") = 10);
}