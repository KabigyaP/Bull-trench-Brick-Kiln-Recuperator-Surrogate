
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy.stats import qmc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

keras.utils.set_random_seed(7)   # same random numbers every run, so results can be repeated


# Fixed design values
wall = 0.0025         # tube wall thickness in m (ASSUMPTION)
k_wall = 45.0         # thermal conductivity of mild steel, W/m.K
pitch_ratio = 1.25    # square tube pitch divided by tube outer diameter
bundle_fill = 0.85    # how much of the shell area the tube layout uses
rf_gas = 3e-4         # soot fouling on the gas side, m2.K/W (ASSUMPTION)
rf_air = 1e-4         # fouling on the air side, m2.K/W (ASSUMPTION)
gas_cp_factor = 1.05  # flue gas holds a bit more heat than air, so cp is scaled up (ASSUMPTION)
p_atm = 101325.0      # pressure in Pa
r_air = 287.0         # gas constant of air, J/kg.K

# Air properties from a textbook table (300 K to 700 K).
# They change with temperature, so we fit a quadratic curve through each one.
T_table = np.array([300, 350, 400, 450, 500, 550, 600, 650, 700], float)
cp_table = np.array([1007, 1009, 1014, 1021, 1030, 1040, 1051, 1063, 1075], float)               # J/kg.K
mu_table = np.array([184.6, 208.2, 230.1, 250.7, 270.1, 288.4, 305.8, 322.5, 338.8]) * 1e-7      # Pa.s
k_table = np.array([26.3, 30.0, 33.8, 37.3, 40.7, 43.9, 46.9, 49.7, 52.4]) * 1e-3                # W/m.K
cp_fit = np.polyfit(T_table, cp_table, 2)
mu_fit = np.polyfit(T_table, mu_table, 2)
k_fit = np.polyfit(T_table, k_table, 2)


def air_properties(T_c):
    # gives cp, viscosity, conductivity, density and Prandtl number of air at T_c (deg C)
    T = T_c + 273.15
    cp = np.polyval(cp_fit, T)
    mu = np.polyval(mu_fit, T)
    k = np.polyval(k_fit, T)
    rho = p_atm / (r_air * T)
    Pr = mu * cp / k
    return cp, mu, k, rho, Pr


def solve_recuperator(Tgi, m_g, m_a, Tai, di, N, L, b_ratio):
    # Takes one design and returns air outlet temp, gas outlet temp, both pressure drops
    # and the two velocities. Tgi/Tai are inlet temps (gas/air), m_g/m_a are flow rates (kg/s),
    # di is tube inner diameter, N is number of tubes, L is tube length,
    # b_ratio is baffle spacing divided by shell diameter.

    # Geometry
    do = di + 2 * wall                                         # tube outer diameter
    pt = pitch_ratio * do                                      # tube pitch
    Ds = np.sqrt(4 * N * pt**2 / (np.pi * bundle_fill))        # shell inside diameter
    B = b_ratio * Ds                                           # baffle spacing
    n_baffles = max(int(L / B) - 1, 0)                         # baffles must be a whole number
    de = 4 * (pt**2 - np.pi * do**2 / 4) / (np.pi * do)        # Kern equivalent diameter
    a_shell = Ds * B * (pt - do) / pt                          # shell side flow area
    A_o = np.pi * do * L * N                                   # outer heat transfer area

    # First guess for the outlet temperatures. The properties depend on temperature and the
    # temperature depends on the properties, so we repeat the calculation until it settles.
    Tgo = Tgi - 0.5 * (Tgi - Tai)
    Tao = Tai + 0.5 * (Tgi - Tai)
    for _ in range(30):
        cp_g, mu_g, k_g, rho_g, Pr_g = air_properties(0.5 * (Tgi + Tgo))   # gas properties at mean temp
        cp_g = cp_g * gas_cp_factor
        cp_a, mu_a, k_a, rho_a, Pr_a = air_properties(0.5 * (Tai + Tao))   # air properties at mean temp

        # Tube side (gas): Reynolds number, friction factor and Nusselt number (Gnielinski)
        Re_t = 4 * m_g / (np.pi * di * mu_g * N)
        f_t = (0.79 * np.log(max(Re_t, 3000)) - 1.64) ** -2
        if Re_t < 2300:
            Nu_t = 3.66    # laminar flow
        else:
            Nu_t = (f_t / 8) * (Re_t - 1000) * Pr_g / (1 + 12.7 * np.sqrt(f_t / 8) * (Pr_g ** (2 / 3) - 1))
        h_t = Nu_t * k_g / di

        # Shell side (air): Kern method
        Gs = m_a / a_shell
        Re_s = Gs * de / mu_a
        Nu_s = 0.36 * Re_s**0.55 * Pr_a ** (1 / 3)
        h_s = Nu_s * k_a / de

        # Overall heat transfer coefficient U (based on outer area).
        # Resistances in series: gas film + gas fouling + tube wall + air fouling + air film
        inv_U = (do / di) * (1 / h_t + rf_gas) + do * np.log(do / di) / (2 * k_wall) + rf_air + 1 / h_s
        U = 1 / inv_U

        # Effectiveness-NTU method for counter-flow
        C_g = m_g * cp_g                       # heat capacity rate of gas
        C_a = m_a * cp_a                       # heat capacity rate of air
        C_min = min(C_g, C_a)
        C_max = max(C_g, C_a)
        Cr = C_min / C_max
        NTU = U * A_o / C_min
        if abs(1 - Cr) < 1e-9:
            eff = NTU / (1 + NTU)
        else:
            e = np.exp(-NTU * (1 - Cr))
            eff = (1 - e) / (1 - Cr * e)
        Q = eff * C_min * (Tgi - Tai)          # heat transferred in W

        Tgo_new = Tgi - Q / C_g
        Tao_new = Tai + Q / C_a
        done = abs(Tao_new - Tao) < 1e-4 and abs(Tgo_new - Tgo) < 1e-4
        Tgo, Tao = Tgo_new, Tao_new
        if done:
            break

    # Pressure drops
    v_t = 4 * m_g / (rho_g * np.pi * di**2 * N)                            # gas velocity in tubes
    dP_g = (f_t * L / di + 1.5) * rho_g * v_t**2 / 2                        # friction + entry/exit losses
    f_s = np.exp(0.576 - 0.19 * np.log(max(Re_s, 400)))                     # Kern shell side friction factor
    dP_a = f_s * Gs**2 * Ds * (n_baffles + 1) / (2 * rho_a * de)            # air side pressure drop
    v_s = Gs / rho_a                                                        # air velocity across the tubes

    return Tao, Tgo, dP_g, dP_a, v_t, v_s


# Quick check on one design before trusting the solver
Tao, Tgo, dP_g, dP_a, v_t, v_s = solve_recuperator(350, 1.5, 1.2, 25, 0.040, 100, 3.0, 0.6)
print('Check design: air out {:.1f} C, gas out {:.1f} C, dP gas {:.0f} Pa, dP air {:.0f} Pa'.format(Tao, Tgo, dP_g, dP_a))

# Part 2: Making the dataset

# The 8 inputs and the range of each one: (name, lowest, highest, use log scale?)
# Log scale is used when the range is very wide, so small values get fair coverage too.
design_space = [
    ('T_gas_in',     200,   450,   False),   # flue gas inlet temperature, deg C
    ('m_gas',        0.5,   3.0,   True),    # flue gas flow, kg/s
    ('air_ratio',    0.70,  1.00,  False),   # air flow divided by gas flow
    ('T_air_in',     10,    35,    False),   # air inlet temperature, deg C
    ('d_in',         0.025, 0.060, True),    # tube inner diameter, m
    ('N_tubes',      40,    200,   True),     # number of tubes
    ('length',       1.5,   5.0,   True),    # tube length, m
    ('baffle_ratio', 0.3,   1.0,   True),    # baffle spacing / shell diameter
]
num_inputs = len(design_space)
n_candidates = 30000

# Latin hypercube sampling spreads the designs evenly over the whole space
# (random picking would leave gaps and clumps)
u = qmc.LatinHypercube(d=num_inputs, seed=7).random(n_candidates)
designs = np.zeros((n_candidates, num_inputs))
for j, (name, low, high, use_log) in enumerate(design_space):
    if use_log:
        designs[:, j] = np.exp(np.log(low) + u[:, j] * (np.log(high) - np.log(low)))
    else:
        designs[:, j] = low + u[:, j] * (high - low)
designs[:, 5] = np.round(designs[:, 5])    # number of tubes must be a whole number

# Run the solver on every design
outputs = np.zeros((n_candidates, 6))
for i in range(n_candidates):
    Tgi, m_g, ratio, Tai, di, N, L, b = designs[i]
    outputs[i] = solve_recuperator(Tgi, m_g, m_g * ratio, Tai, di, int(N), L, b)
# columns of outputs: 0 air out temp, 1 gas out temp, 2 gas dP, 3 air dP, 4 tube velocity, 5 air velocity

# Keep only designs an engineer would accept:
# gas velocity 5 to 30 m/s, air velocity under 25 m/s, each pressure drop under 3000 Pa
ok = (outputs[:, 4] > 5) & (outputs[:, 4] < 30) & (outputs[:, 5] < 25) & (outputs[:, 2] < 3000) & (outputs[:, 3] < 3000)
designs = designs[ok][:7000]
outputs = outputs[ok][:7000]
print('Designs kept:', ok.sum(), 'out of', n_candidates, '(using', len(designs), ')')
print(designs.shape, outputs.shape)

# Look at the data first: how are the four results spread?
labels = ['Air outlet temp (C)', 'Gas outlet temp (C)', 'Gas pressure drop (Pa)', 'Air pressure drop (Pa)']
f, ax = plt.subplots(1, 4, figsize=(20, 4))
for k in range(4):
    ax[k].hist(outputs[:, k], bins=40)
    ax[k].set_title(labels[k], fontsize=14)
plt.savefig('data_overview.png', dpi=100)
plt.show()

# Part 3: Preparing the data for the network

# Split first, so the test designs stay completely hidden until the very end
idx = np.arange(len(designs))
train_idx, test_idx = train_test_split(idx, test_size=0.15, random_state=7)

# Baffle count: the air crosses the tube bundle a whole number of times, so the air pressure
# drop jumps in steps. A smooth network cannot guess these steps, so we calculate the count
# ourselves and give it to the network as an extra input.
d_in, N_t, length, b_ratio = designs[:, 4], designs[:, 5], designs[:, 6], designs[:, 7]
pt = pitch_ratio * (d_in + 2 * wall)
Ds = np.sqrt(4 * N_t * pt**2 / (np.pi * bundle_fill))
baffle_count = np.maximum((length / (b_ratio * Ds)).astype(int) - 1, 0) + 1

# Inputs. Most heat exchanger relations are power laws, and taking the log turns them into
# straight lines, which are much easier to learn. Temperatures stay as they are.
x_all = np.column_stack([
    designs[:, 0],              # gas inlet temp
    np.log(designs[:, 1]),      # gas flow
    np.log(designs[:, 2]),      # air ratio
    designs[:, 3],              # air inlet temp
    np.log(designs[:, 4]),      # tube diameter
    np.log(designs[:, 5]),      # number of tubes
    np.log(designs[:, 6]),      # tube length
    np.log(designs[:, 7]),      # baffle ratio
    np.log(baffle_count),       # baffle count
])

# Targets. Instead of raw temperatures we predict fractions between 0 and 1:
# how much of the possible air heating we got, and how much the gas cooled.
# Pressure drops cover a huge range, so we predict their log.
Tgi_all, Tai_all = designs[:, 0], designs[:, 3]
span = Tgi_all - Tai_all
y_all = np.column_stack([
    (outputs[:, 0] - Tai_all) / span,       # air side effectiveness
    (Tgi_all - outputs[:, 1]) / span,       # gas cooling fraction
    np.log(outputs[:, 2]),                  # log gas pressure drop
    np.log(outputs[:, 3]),                  # log air pressure drop
])

x_train, x_test = x_all[train_idx], x_all[test_idx]
y_train, y_test = y_all[train_idx], y_all[test_idx]

# Standardization: bring every column to mean 0 and standard deviation 1.
# The mean and std come from the training data only, so the test data stays unseen.
x_mean, x_std = x_train.mean(axis=0), x_train.std(axis=0)
y_mean, y_std = y_train.mean(axis=0), y_train.std(axis=0)
x_train_s = (x_train - x_mean) / x_std
x_test_s = (x_test - x_mean) / x_std
y_train_s = (y_train - y_mean) / y_std
print(x_train_s.shape, y_train_s.shape, x_test_s.shape)


def to_real_units(y_scaled, idx):
    # turns the network output back into temperatures (C) and pressure drops (Pa)
    y = y_scaled * y_std + y_mean
    Tgi, Tai = designs[idx, 0], designs[idx, 3]
    sp = Tgi - Tai
    return np.column_stack([Tai + y[:, 0] * sp, Tgi - y[:, 1] * sp, np.exp(y[:, 2]), np.exp(y[:, 3])])


def print_errors(pred, true, title):
    # average error in deg C for the temperatures and in % for the pressure drops
    mae_air = np.mean(np.abs(pred[:, 0] - true[:, 0]))
    mae_gas = np.mean(np.abs(pred[:, 1] - true[:, 1]))
    pct_g = 100 * np.mean(np.abs(pred[:, 2] - true[:, 2]) / true[:, 2])
    pct_a = 100 * np.mean(np.abs(pred[:, 3] - true[:, 3]) / true[:, 3])
    print('{}: air out {:.2f} C, gas out {:.2f} C, gas dP {:.2f} %, air dP {:.2f} %'.format(title, mae_air, mae_gas, pct_g, pct_a))


true_test = outputs[test_idx, :4]    # the solver's real answers for the test designs

# Part 4: Baseline (to see if the neural network is actually worth it)
# A plain linear model on the same inputs. If the network can't beat this, it isn't needed.
linear = LinearRegression().fit(x_train_s, y_train_s)
print_errors(to_real_units(linear.predict(x_test_s), test_idx), true_test, 'Linear model')


# Part 5: Building the neural network

epochs = 600       # at most 600 runs through the data (early stopping can end it sooner)
batch_size = 64    # learn from 64 designs at a time

model = Sequential()
model.add(Dense(128, activation='tanh'))   # first layer: 128 neurons. tanh is smooth, and this problem is smooth
model.add(Dense(128, activation='tanh'))   # second layer: 128 neurons
model.add(Dense(64, activation='tanh'))    # third layer: 64 neurons
model.add(Dense(4))                        # output layer: 4 numbers, no activation because this is regression
# No dropout here. The data comes from an exact solver with no noise, so there is nothing to memorize wrongly.

# mse (mean squared error) is the usual loss for predicting numbers
# adam updates the weights, mae is printed so we can read the error easily
model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.002), metrics=['mae'])

# Early stopping: watch the error on validation data (15% of the training set, set aside by Keras).
# When it stops improving for 60 epochs, stop and go back to the best weights. This prevents overfitting.
stop_early = EarlyStopping(monitor='val_loss', patience=60, restore_best_weights=True)
# Learning rate reduction: when progress stalls for 15 epochs, take smaller steps (half the size).
# Big steps are good at the start, small steps help to fine tune at the end.
slow_down = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-5)

# Training the model
history = model.fit(x_train_s, y_train_s, epochs=epochs, batch_size=batch_size,
                    validation_split=0.15, callbacks=[stop_early, slow_down], verbose=2)
print('Stopped after', len(history.history['loss']), 'epochs')

# Plot the training curves: if the two lines stay close, the model is not overfitting
plt.figure(figsize=(7, 4))
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.yscale('log')
plt.xlabel('epoch')
plt.ylabel('loss (mse)')
plt.legend()
plt.savefig('training_curve.png', dpi=100)
plt.show()

# Part 6: Testing the model on designs it has never seen

pred_test = to_real_units(model.predict(x_test_s, verbose=0), test_idx)
print_errors(pred_test, true_test, 'Neural network')

# Predicted vs solver. Points on the dashed line mean a perfect match.
f, ax = plt.subplots(1, 4, figsize=(20, 5))
use_log = [False, False, True, True]
for k in range(4):
    ax[k].scatter(true_test[:, k], pred_test[:, k], s=5, alpha=0.5)
    lo = min(true_test[:, k].min(), pred_test[:, k].min())
    hi = max(true_test[:, k].max(), pred_test[:, k].max())
    ax[k].plot([lo, hi], [lo, hi], 'k--')
    if use_log[k]:
        ax[k].set_xscale('log')
        ax[k].set_yscale('log')
    ax[k].set_xlabel('Solver')
    ax[k].set_ylabel('Neural network')
    ax[k].set_title(labels[k], fontsize=14)
plt.savefig('predicted_vs_solver.png', dpi=100)
plt.show()

# Compare the network and the solver on the first 3 test designs (never used for training)
for i in range(3):
    n_air, n_gas, n_dpg, n_dpa = pred_test[i]
    s_air, s_gas, s_dpg, s_dpa = true_test[i]
    print('Design {}: network air out {:.1f} C, gas out {:.1f} C, dP gas {:.0f} Pa, dP air {:.0f} Pa'.format(i, n_air, n_gas, n_dpg, n_dpa))
    print('          solver  air out {:.1f} C, gas out {:.1f} C, dP gas {:.0f} Pa, dP air {:.0f} Pa'.format(s_air, s_gas, s_dpg, s_dpa))
