clc; clear; close all;

% Initialize time-related parameters
tic
T = 2; % try 1 
N_t = 20000; % try 10000 or 100000
dt = T/N_t;
t = 0:dt:T;

% Initialize parameters for the mesh
x_min = 0; x_max = 1; y_min = 0; y_max = 1;
N_x = 100; N_y = N_x;  % Mesh size
h_x = (x_max-x_min)/(N_x-1);
h_y = (y_max-y_min)/(N_y-1);
x = linspace(x_min, x_max, N_x);
y = linspace(y_min, y_max, N_y);
[X, Y] = meshgrid(x, y);

% Parameters estimated based off typical literature values
% B = 1.0e-11; or e-8
% kd2 = 1.7e-9; or e-6
B = 1.0e-8;
kd2 = 1.7e-6;
Omega = -(4 + h_x^2*kd2/2);

% Initialize streamfunctions
psi_1(:,:,1) = sin(3*pi*dt*(X-(X.^2))).*cos(2*pi*dt*(Y-(Y.^2))).*sin(5*pi*dt*(Y-(Y.^2))).*cos(6*pi*dt*(X-(X.^2)));
psi_2(:,:,1) = sin(5*pi*dt*(X-(X.^2))).*cos(pi*dt*(Y-(Y.^2))).*sin(3*pi*dt*(Y-(Y.^2))).*cos(2*pi*dt*(X-(X.^2)));
% psi_1(:,:,1) = sin(2*pi*dt*X).*cos(2*pi*dt*Y);
% psi_2(:,:,1) = sin(4*pi*dt*X).*cos(2*pi*dt*Y);

% psi_1(:,:,1) = -sin(1.2 * dt * pi * X) .* sin(1.6 * dt * pi * Y) + 0.6 * cos(2.4 * dt * pi * X) .* cos(2.8 * dt * pi * Y);
% psi_2(:,:,1) = sin(3.2 * dt * pi * X) .* sin(0.8 * dt * pi * Y) + 0.7 * cos(1.6 * dt * pi * X) .* cos(2.4 * dt * pi * Y);

syms a b c % t = a, x = b, y = c
% psi_1_test = sin(3*pi*a*(b-(b.^2))).*cos(2*pi*a*(c-(c.^2))).*sin(5*pi*a*(c-(c.^2))).*cos(6*pi*a*(b-(b.^2)));
% psi_2_test = sin(5*pi*a*(b-(b.^2))).*cos(pi*a*(c-(c.^2))).*sin(3*pi*a*(c-(c.^2))).*cos(2*pi*a*(b-(b.^2)));
psi_1_test = -sin(1.2 * pi * a * b) .* sin(1.6 * pi * a * c) + 0.6 * cos(2.4 * pi * a * b) .* cos(2.8 * pi * a * c);
psi_2_test = sin(3.2 * pi * a * b) .* sin(0.8 * pi * a * c) + 0.7 * cos(1.6 * pi * a * b) .* cos(2.4 * pi * a * c);

psi_1_dx = diff(psi_1_test,b);
psi_1_dy = diff(psi_1_test,c);
psi_1_dxx = diff(psi_1_dx,b);
psi_1_dyy = diff(psi_1_dy,c);
psi_1_lap = psi_1_dxx + psi_1_dyy;

psi_2_dx = diff(psi_2_test,b);
psi_2_dy = diff(psi_2_test,c);
psi_2_dxx = diff(psi_2_dx,b);
psi_2_dyy = diff(psi_2_dy,c);
psi_2_lap = psi_2_dxx + psi_2_dyy;

q_1_test = psi_1_lap + B*c + (kd2/2)*(psi_2_test - psi_1_test);
q_2_test = psi_2_lap + B*c + (kd2/2)*(psi_1_test - psi_2_test);

F_1 = diff(q_1_test, a) + psi_1_dx .* diff(q_1_test, c) - psi_1_dy .* diff(q_1_test, b);
F_2 = diff(q_2_test, a) + psi_2_dx .* diff(q_2_test, c) - psi_2_dy .* diff(q_2_test, b);

F_1_func = matlabFunction(F_1, 'Vars', {a, b, c});
F_2_func = matlabFunction(F_2, 'Vars', {a, b, c});

psi_1_func = matlabFunction(psi_1_test, 'Vars', {a,b,c});
psi_2_func = matlabFunction(psi_2_test, 'Vars', {a,b,c});

q_1_func = matlabFunction(q_1_test, 'Vars', {a,b,c});
q_2_func = matlabFunction(q_2_test, 'Vars', {a,b,c});

% Initialize q for each layer
q_1 = zeros(N_x, N_y, N_t);
q_2 = zeros(N_x, N_y, N_t);

% Initialize q for t=0 using equation (2)
q_1(:,:,1) = del2(psi_1(:,:,1), h_x, h_y) + B * Y + (kd2 / 2) * (psi_2(:,:,1) - psi_1(:,:,1));
q_2(:,:,1) = del2(psi_2(:,:,1), h_x, h_y) + B * Y + (kd2 / 2) * (psi_1(:,:,1) - psi_2(:,:,1));

% Construct A using the provided function
A = constructA(N_x, N_y, Omega);
disp(cond(A));
disp(det(A));
toc
A_inv = A\eye(size(A));
toc
%%
% Time-stepping loop
for m = 1:N_t-1
    % Update stream function derivatives for each layer
    % Layer 1
    [dpsi_1_dx, dpsi_1_dy] = gradient(psi_1(:,:,m), h_x, h_y);
    [dq_1_dx, dq_1_dy] = gradient(q_1(:,:,m), h_x, h_y);

    % Layer 2
    [dpsi_2_dx, dpsi_2_dy] = gradient(psi_2(:,:,m), h_x, h_y);
    [dq_2_dx, dq_2_dy] = gradient(q_2(:,:,m), h_x, h_y);

    term_1_1 = dpsi_1_dy .* dq_1_dx;
    term_2_1 = dpsi_1_dx .* dq_1_dy;

    F_1_res = F_1_func((m)*dt,X,Y);
    q_1(:,:, m+1) = q_1(:,:, m) + dt * (F_1_res + term_1_1 - term_2_1);

    term_1_2 = dpsi_2_dy .* dq_2_dx;
    term_2_2 = dpsi_2_dx .* dq_2_dy;

    F_2_res = F_2_func(m*dt,X,Y);
    q_2(:,:, m+1) = q_2(:,:, m) + dt * (F_2_res + term_1_2 - term_2_2);

    % For Step 2 (updating psi) by inverting the matrix equation for psi
    % Construct the right-hand side vector for layer 1 and layer 2
    idx1 = 1:N_x; idx2 = N_x+1: N_x: N_x*N_y; idx3 = N_x*N_y:-1:N_x*(N_y - 1) + 1; idx4 = N_x:N_x:N_x*N_y;
    C_1 = h_x^2 * reshape(q_1(:,:,m+1) - B * Y - (kd2 / 2) * (psi_2(:,:,m)), [], 1);
    C_1(idx1) = 0; C_1(idx2) = 0; C_1(idx3) = 0; C_1(idx4) = 0;
    C_2 = h_x^2 * reshape(q_2(:,:,m+1) - B * Y - (kd2 / 2) * (psi_1(:,:,m)), [], 1);
    C_2(idx1) = 0; C_2(idx2) = 0; C_2(idx3) = 0; C_2(idx4) = 0;
    % Solve for psi_1 and psi_2 at the current time step
    psi_1_vector = A_inv * C_1; 
    psi_2_vector = A_inv * C_2;

    % Reshape the solution back into matrix form
    psi_1(:,:,m+1) = reshape(psi_1_vector, N_x, N_y);
    psi_2(:,:,m+1) = reshape(psi_2_vector, N_x, N_y);
end
toc

%% Visualization
% Predefined times for the slices
time_indices = [10000, 13000, 16000, 19000];  % Example indices corresponding to t = 0.1, 0.3, 0.5, 0.7

% Precompute analytical results for all selected time indices
analytical_q1_results = zeros(N_x, N_y, length(time_indices));
analytical_q2_results = zeros(N_x, N_y, length(time_indices));
analytical_psi1_results = zeros(N_x, N_y, length(time_indices));
analytical_psi2_results = zeros(N_x, N_y, length(time_indices));

for i = 1:length(time_indices)
    t_val = time_indices(i) * dt;
    analytical_q1_results(:, :, i) = double(subs(q_1_test, {a, b, c}, {t_val, X, Y}));
    analytical_q2_results(:, :, i) = double(subs(q_2_test, {a, b, c}, {t_val, X, Y}));
    analytical_psi1_results(:, :, i) = double(subs(psi_1_test, {a, b, c}, {t_val, X, Y}));
    analytical_psi2_results(:, :, i) = double(subs(psi_2_test, {a, b, c}, {t_val, X, Y}));
end

figure;
for i = 1:length(time_indices)
    idx = time_indices(i);
    t_val = idx * dt;

    % Using precomputed analytical results
    analytical_q1 = analytical_q1_results(:, :, i);
    analytical_q2 = analytical_q2_results(:, :, i);

    % Retrieve numerical results
    numerical_q1 = reshape(q_1(:, :, idx), [N_x, N_y]);
    numerical_q2 = reshape(q_2(:, :, idx), [N_x, N_y]);

    % Plotting
    subplot(4, 4, i);
    imagesc(x, y, numerical_q1);
    colorbar; axis equal tight;
    title(sprintf('Numerical q1: t = %.1f', t_val));

    subplot(4, 4, i+4);
    imagesc(x, y, analytical_q1);
    colorbar; axis equal tight;
    title(sprintf('Analytical q1: t = %.1f', t_val));

    subplot(4, 4, i+8);
    imagesc(x, y, numerical_q2);
    colorbar; axis equal tight;
    title(sprintf('Numerical q2: t = %.1f', t_val));

    subplot(4, 4, i+12);
    imagesc(x, y, analytical_q2);
    colorbar; axis equal tight;
    title(sprintf('Analytical q2: t = %.1f', t_val));
end
toc

%% RMSE and std Computation
rmse_q1 = zeros(size(time_indices));
rmse_q2 = zeros(size(time_indices));
rmse_psi1 = zeros(size(time_indices));
rmse_psi2 = zeros(size(time_indices));

% Create figures and compute RMSE and standard deviations
disp('Detailed Results for Each Time Index:');
for i = 1:length(time_indices)
    idx = time_indices(i);
    t_val = idx * dt;

    % Retrieve precomputed analytical results
    analytical_q1 = analytical_q1_results(:, :, i);
    analytical_q2 = analytical_q2_results(:, :, i);
    analytical_psi1 = analytical_psi1_results(:, :, i);
    analytical_psi2 = analytical_psi2_results(:, :, i);

    % Retrieve numerical results
    numerical_q1 = reshape(q_1(:, :, idx), [N_x, N_y]);
    numerical_q2 = reshape(q_2(:, :, idx), [N_x, N_y]);
    numerical_psi1 = reshape(psi_1(:, :, idx), [N_x, N_y]);
    numerical_psi2 = reshape(psi_2(:, :, idx), [N_x, N_y]);

    % Calculate RMSE for each dataset
    rmse_q1(i) = sqrt(mean((numerical_q1(:) - analytical_q1(:)).^2));
    rmse_q2(i) = sqrt(mean((numerical_q2(:) - analytical_q2(:)).^2));
    rmse_psi1(i) = sqrt(mean((numerical_psi1(:) - analytical_psi1(:)).^2));
    rmse_psi2(i) = sqrt(mean((numerical_psi2(:) - analytical_psi2(:)).^2));

    % Calculate standard deviations for each dataset
    std_q1 = std(analytical_q1(:));
    std_q2 = std(analytical_q2(:));
    std_psi1 = std(analytical_psi1(:));
    std_psi2 = std(analytical_psi2(:));

    % Display formatted results for RMSE and standard deviation
    fprintf('\nTime index %d: t = %.2f\n', i, t_val);
    fprintf('    RMSE: q1=%.4f, q2=%.4f, psi1=%.4f, psi2=%.4f\n', rmse_q1(i), rmse_q2(i), rmse_psi1(i), rmse_psi2(i));
    fprintf('    Std:  q1=%.4f, q2=%.4f, psi1=%.4f, psi2=%.4f\n', std_q1, std_q2, std_psi1, std_psi2);
end
toc

%%
function A = constructA(N_x, N_y, Omega)
    % Total number of grid points
    N = N_x * N_y;
    
    % Initialize A with zeros
    A = zeros(N, N);
    
    % Helper function to convert 2D indices to 1D index
    idx = @(x, y) (mod(x-1, N_x)) * N_y + mod(y-1, N_y) + 1;
    
    for x = 1:N_x
        for y = 1:N_y
            % Current position index
            currentIdx = idx(x, y);
            
            % Set the diagonal value
            A(currentIdx, currentIdx) = Omega;
            
            % Adjacent indices (accounting for periodic boundary conditions)
            leftIdx = idx(x-1, y);
            rightIdx = idx(x+1, y);
            upIdx = idx(x, y-1);
            downIdx = idx(x, y+1);
            
            % Set the adjacent values to 1
            A(currentIdx, [leftIdx, rightIdx, upIdx, downIdx]) = 1;
        end
    end
    
    idx1 = 1:N_x; idx2 = N_x+1: N_x: N_x*N_y; idx3 = N_x*N_y:-1:N_x*(N_y - 1) + 1; idx4 = N_x:N_x:N_x*N_y;

    A(idx1, :) = 0; 
    for j = 1:length(idx1)
        A(idx1(j),idx1(j)) = 1;
    end
   
    A(idx2, :) = 0;
    for j = 1:length(idx2)
        A(idx2(j),idx2(j)) = 1;
    end

    A(idx3, :) = 0;
    for j = 1:length(idx3)
        A(idx3(j),idx3(j)) = 1;
    end

    A(idx4, :) = 0;
    for j = 1:length(idx4)
        A(idx4(j),idx4(j)) = 1;
    end
 
end
