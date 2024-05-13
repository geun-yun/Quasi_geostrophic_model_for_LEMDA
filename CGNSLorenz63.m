tic
% Lorenz system parameters
sigma = 10; rho = 28; beta = 8/3;

% Time settings
dt = 0.01; T = 20;
tSpan = 0:dt:T;

% Initial conditions
X = 5;  % State variable 'x'
Y = [5; 5];  % State variables 'y' and 'z'
mu_f = Y;  % Initial conditional mean of Y
R_f = eye(2) *1.4;  % Initial conditional covariance of Y

% Stochasticity parameters
sigma_x = sqrt(2);  % Noise intensity for X
sigma_y = sqrt(12); sigma_z = sqrt(12);  % Noise intensities for Y components

% Define A_0, A_1, a_0, a_1 (inside the loop as X changes), B_1, and b_2
B_1 = sigma_x;  % Stochastic influence on X
b_2 = [sigma_y, 0; 0, sigma_z];  % Stochastic influences on Y

% Initialize history arrays
X_hist = zeros(1, length(tSpan));
Y_hist = zeros(2, length(tSpan));
mu_f_hist = zeros(2, length(tSpan));

for i = 1:length(tSpan)
    t = tSpan(i);
    
    A_0 = -sigma * X;  % A_0 for the X dynamics
    A_1 = [sigma, 0];  % A_1 maps influence of Y on X
    a_0 = [rho * X; 0];  % a_0 for the Y dynamics, dependent on X
    a_1 = [-1, -X; X, -beta];  % a_1 for the Y dynamics

    % Wiener process increments
    dW1 = sqrt(dt) * randn;  % For X
    dW2 = sqrt(dt) * randn(2, 1);  % For Y

    % Equation 8.1a: Update X using its dynamics
    dX = (A_0 + A_1 * Y) * dt + B_1 * dW1;
    X = X + dX;

    % Equation 8.1b: Update Y using its dynamics
    dY = (a_0 + a_1 * Y) * dt + b_2 * dW2;
    Y = Y + dY;

    % Store history
    X_hist(i) = X;
    Y_hist(:, i) = Y;

    % Equation 8.8a: Update mu_f 
    dmu_f = (a_0 + a_1*mu_f)*dt + (1/B_1^2)*(R_f*A_1')*(dX - (A_0 + A_1*mu_f)*dt);
    mu_f = mu_f + dmu_f;
    
    %Equation 8.8b: Update R_f
    dR_f = (a_1*R_f + R_f*a_1' + b_2*b_2' - (1/B_1^2)*(R_f*A_1')*(A_1*R_f))*dt;
    R_f = R_f + dR_f;
    
    % Store history of conditional mean
    mu_f_hist(:, i) = mu_f;
end

% Plot results
figure;
subplot(2, 1, 1);
plot(tSpan, Y_hist(1, :), 'r', tSpan, mu_f_hist(1, :), 'b--');
legend('True y', 'Estimated y');
title('Y and Conditional Mean (\mu_f) for y-component');

subplot(2, 1, 2);
plot(tSpan, Y_hist(2, :), 'r', tSpan, mu_f_hist(2, :), 'b--');
legend('True z', 'Estimated z');
title('Y and Conditional Mean (\mu_f) for z-component');
toc