clc; clear; close all;

% Initialize time-related parameters
tic;
T = 2; % Total simulation time
N_t = 20000; % Number of time steps
dt = T / N_t; % Time step size
t = 0:dt:T;

% Initialize parameters for the mesh
x_min = 0; x_max = 1; y_min = 0; y_max = 1;
N_x = 50; N_y = N_x; % Mesh size
h_x = (x_max - x_min) / (N_x - 1);
h_y = (y_max - y_min) / (N_y - 1);
x = linspace(x_min, x_max, N_x);
y = linspace(y_min, y_max, N_y);
[X, Y] = meshgrid(x, y);

% QG parameters
B = 0.1; % drag coefficient
kd2 = sqrt(2);
Omega = -(4 + h_x^2 * kd2 / 2);
F = 0;

% Initialize stream functions
%psi_1(:,:,1) = -sin(1.2 * pi * X) .* sin(1.6 * pi * Y) + 0.6 * cos(2.4 * pi * X) .* cos(2.8 * pi * Y);
%psi_2(:,:,1) = sin(3.2 * pi * X) .* sin(0.8 * pi * Y) + 0.7 * cos(1.6 * pi * X) .* cos(2.4 * pi * Y);
psi_1(:,:,1) = exp(-(2 * (X - 1/2).^2 + (Y - 1/2).^2) / (2 * (1 / 8)^2));
psi_2(:,:,1) = exp(-((X - 1/2).^2 + 4 * (Y - 1/2).^2) / (3 * (1 / 8)^2));

% Initialize q for each layer
q_1 = zeros(N_x, N_y, N_t);
q_2 = zeros(N_x, N_y, N_t);

% Initialize q for t = 0 using equation (2)
q_1(:,:,1) = del2(psi_1(:,:,1), h_x, h_y) + B * Y + (kd2 / 2) * (psi_2(:,:,1) - psi_1(:,:,1));
q_2(:,:,1) = del2(psi_2(:,:,1), h_x, h_y) + B * Y + (kd2 / 2) * (psi_1(:,:,1) - psi_2(:,:,1));

% Initialize storage for velocity fields
u_field_1 = zeros(N_x, N_y, N_t);
v_field_1 = zeros(N_x, N_y, N_t);
u_field_2 = zeros(N_x, N_y, N_t);
v_field_2 = zeros(N_x, N_y, N_t);

% Construct A using the provided function
A = constructA(N_x, N_y, Omega);
A_inv = A \ eye(size(A));
disp(cond(A));
disp(det(A));
toc;

% Particle tracing parameters
numParticles = 10; % Number of particles to trace

% Initial particle positions for both layers (random)
% 0.3 and 0.7 (should be x_min and x_max) set as the range of coordinates for the Gaussian 
% since other coordinates don't really capture the dynamic
particles_x_1 = rand(1, numParticles) * (0.7 - 0.3) + 0.3;
particles_y_1 = rand(1, numParticles) * (0.7 - 0.3) + 0.3;
particles_x_2 = rand(1, numParticles) * (0.7 - 0.3) + 0.3;
particles_y_2 = rand(1, numParticles) * (0.7 - 0.3) + 0.3;

% Initialize particle velocities (starting with zero velocity)
particles_vx_1 = zeros(1, numParticles);
particles_vy_1 = zeros(1, numParticles);
particles_vx_2 = zeros(1, numParticles);
particles_vy_2 = zeros(1, numParticles);

% Store particle trajectories over time for both layers
particle_trajectories_x_1 = zeros(N_t, numParticles);
particle_trajectories_y_1 = zeros(N_t, numParticles);
particle_trajectories_x_2 = zeros(N_t, numParticles);
particle_trajectories_y_2 = zeros(N_t, numParticles);
particle_trajectories_x_1(1, :) = particles_x_1;
particle_trajectories_y_1(1, :) = particles_y_1;
particle_trajectories_x_2(1, :) = particles_x_2;
particle_trajectories_y_2(1, :) = particles_y_2;

% Function to find the velocity at a given position using interpolation
interpolateVelocity = @(grid, u_field, v_field, x_pos, y_pos) ...
    deal(interp2(grid.x, grid.y, u_field, x_pos, y_pos, 'linear', 0), ...
         interp2(grid.x, grid.y, v_field, x_pos, y_pos, 'linear', 0));

%%
% Time-stepping loop
for m = 1:N_t-1
    % Update stream function derivatives for Layer 1
    [dpsi_1_dx, dpsi_1_dy] = gradient(psi_1(:,:,m), h_x, h_y);
    [dq_1_dx, dq_1_dy] = gradient(q_1(:,:,m), h_x, h_y);
    u_field_1(:, :, m) = dpsi_1_dy; % u-component of velocity
    v_field_1(:, :, m) = -dpsi_1_dx; % v-component of velocity

    % Update stream function derivatives for Layer 2
    [dpsi_2_dx, dpsi_2_dy] = gradient(psi_2(:,:,m), h_x, h_y);
    [dq_2_dx, dq_2_dy] = gradient(q_2(:,:,m), h_x, h_y);
    u_field_2(:, :, m) = dpsi_2_dy; % u-component of velocity
    v_field_2(:, :, m) = -dpsi_2_dx; % v-component of velocity

    % Update each particle position for Layer 1
    for p = 1:numParticles
        % Obtain velocity components for the current particle position via interpolation
        [u_particle, v_particle] = interpolateVelocity(struct('x', x, 'y', y), u_field_1(:, :, m), v_field_1(:, :, m), particles_x_1(p), particles_y_1(p));

        % Update velocity (Equation 2.1b)
        particles_vx_1(p) = particles_vx_1(p) + dt * (B * (u_particle - particles_vx_1(p)) + F);
        particles_vy_1(p) = particles_vy_1(p) + dt * (B * (v_particle - particles_vy_1(p)) + F);

        % Update position (Equation 2.1a)
        particles_x_1(p) = particles_x_1(p) + (particles_vx_1(p) + 0) * dt;
        particles_y_1(p) = particles_y_1(p) + (particles_vy_1(p) + 0) * dt;

        % Ensure particles remain within the grid
        particles_x_1(p) = mod(particles_x_1(p) - x_min, x_max - x_min) + x_min;
        particles_y_1(p) = mod(particles_y_1(p) - y_min, y_max - y_min) + y_min;
    end

    % Update each particle position for Layer 2
    for p = 1:numParticles
        % Obtain velocity components for the current particle position via interpolation
        [u_particle, v_particle] = interpolateVelocity(struct('x', x, 'y', y), u_field_2(:, :, m), v_field_2(:, :, m), particles_x_2(p), particles_y_2(p));

        % Update velocity (Equation 2.1b)
        particles_vx_2(p) = particles_vx_2(p) + dt * (B * (u_particle - particles_vx_2(p)) + F);
        particles_vy_2(p) = particles_vy_2(p) + dt * (B * (v_particle - particles_vy_2(p)) + F);

        % Update position (Equation 2.1a)
        particles_x_2(p) = particles_x_2(p) + (particles_vx_2(p) + 0) * dt;
        particles_y_2(p) = particles_y_2(p) + (particles_vy_2(p) + 0) * dt;

        % Ensure particles remain within the grid
        particles_x_2(p) = mod(particles_x_2(p) - x_min, x_max - x_min) + x_min;
        particles_y_2(p) = mod(particles_y_2(p) - y_min, y_max - y_min) + y_min;
    end

    % Store particle positions over time
    particle_trajectories_x_1(m+1, :) = particles_x_1;
    particle_trajectories_y_1(m+1, :) = particles_y_1;
    particle_trajectories_x_2(m+1, :) = particles_x_2;
    particle_trajectories_y_2(m+1, :) = particles_y_2;

    % Update q values using the provided method (Step 1)
    term_1_1 = dpsi_1_dy .* dq_1_dx;
    term_2_1 = dpsi_1_dx .* dq_1_dy;
    q_1(:,:, m+1) = q_1(:,:, m) + dt * (F + term_1_1 - term_2_1);

    term_1_2 = dpsi_2_dy .* dq_2_dx;
    term_2_2 = dpsi_2_dx .* dq_2_dy;
    q_2(:,:, m+1) = q_2(:,:, m) + dt * (F + term_1_2 - term_2_2);

    % Apply boundary conditions to q_1 and q_2
    q_1(1,:,m+1) = 0; q_1(:,1,m+1) = 0;
    q_1(end,:,m+1) = 0; q_1(:,end,m+1) = 0;

    q_2(1,:,m+1) = 0; q_2(:,1,m+1) = 0;
    q_2(end,:,m+1) = 0; q_2(:,end,m+1) = 0;

    % Update psi fields using Step 2 (inverting the matrix equation)
    idx1 = 1:N_x; idx2 = N_x+1: N_x: N_x*N_y; idx3 = N_x*N_y:-1:N_x*(N_y-1) + 1; idx4 = N_x:N_x:N_x*N_y;
    C_1 = h_x^2 * reshape(q_1(:,:,m+1) - B * Y - (kd2 / 2) * (psi_2(:,:,m)), [], 1);
    C_1(idx1) = 0; C_1(idx2) = 0; C_1(idx3) = 0; C_1(idx4) = 0;
    C_2 = h_x^2 * reshape(q_2(:,:,m+1) - B * Y - (kd2 / 2) * (psi_1(:,:,m)), [], 1);
    C_2(idx1) = 0; C_2(idx2) = 0; C_2(idx3) = 0; C_2(idx4) = 0;

    % Solve for psi values
    psi_1_vector = A_inv * C_1;
    psi_2_vector = A_inv * C_2;

    % Reshape into matrix form
    psi_1(:,:,m+1) = reshape(psi_1_vector, N_x, N_y);
    psi_2(:,:,m+1) = reshape(psi_2_vector, N_x, N_y);
end
toc;

%%
% Setup video writer
outputDir = 'C:\Users\yje06\Desktop\ㅇㅎㄱ\ANU\2024_Semester_1\COMP3770\QGPV\video\'; % Adjust path as needed
videoFileName = ['QGPV_Particle_Tracing_Nt', num2str(N_t), 'T', num2str(T), '_Nx', num2str(N_x), '.mp4'];
fullVideoPath = [outputDir videoFileName];
v = VideoWriter(fullVideoPath, 'MPEG-4');
v.FrameRate = 5; % Adjust the frame rate as required
skipfreq = 200;
open(v); % Open the video file for writing

% Create a set of distinct colors for different particle paths
colors = lines(numParticles); % Generate an array of distinguishable colors
lineWidth = 2; % Set a thicker line width

hFig = figure('WindowState', 'maximized'); % Create a new figure for video frames
for m = 1:N_t/skipfreq
    % Clear current figure
    figure(hFig);
    clf(hFig);

    % Plot Layer 1 with particle paths
    subplot(1, 2, 1);
    imagesc(x, y, psi_1(:, :, skipfreq * m));
    axis equal tight;
    hold on;
    for p = 1:numParticles
        plot(particle_trajectories_x_1(1:skipfreq * m, p), particle_trajectories_y_1(1:skipfreq * m, p), '-', 'Color', colors(p, :), 'LineWidth', lineWidth);
        plot(particle_trajectories_x_1(skipfreq * m, p), particle_trajectories_y_1(skipfreq * m, p), 'o', 'Color', colors(p, :), 'MarkerFaceColor', 'r', 'MarkerSize', 6);
    end
    hold off;
    colorbar;
    title(sprintf('Layer 1 Particle Tracing at t = %.4f', skipfreq * m * dt));

    % Plot Layer 2 with particle paths
    subplot(1, 2, 2);
    imagesc(x, y, psi_2(:, :, skipfreq * m));
    axis equal tight;
    hold on;
    for p = 1:numParticles
        plot(particle_trajectories_x_2(1:skipfreq * m, p), particle_trajectories_y_2(1:skipfreq * m, p), '-', 'Color', colors(p, :), 'LineWidth', lineWidth);
        plot(particle_trajectories_x_2(skipfreq * m, p), particle_trajectories_y_2(skipfreq * m, p), 'o', 'Color', colors(p, :), 'MarkerFaceColor', 'r', 'MarkerSize', 6);
    end
    hold off;
    colorbar;
    title(sprintf('Layer 2 Particle Tracing at t = %.4f', skipfreq * m * dt));

    drawnow; % Update figure window

    % Capture the frame
    frame = getframe(hFig);
    writeVideo(v, frame); % Write the frame to the video
end
close(v);
toc;

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
