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
F = 1;

% Initialize streamfunctions
% psi_1(:,:,1) = sin(3*pi*(X-(X.^2))).*cos(2*pi*(Y-(Y.^2))).*sin(5*pi*(Y-(Y.^2))).*cos(6*pi*(X-(X.^2)));
% psi_2(:,:,1) = sin(5*pi*(X-(X.^2))).*cos(pi*(Y-(Y.^2))).*sin(3*pi*(Y-(Y.^2))).*cos(2*pi*(X-(X.^2)));
% psi_1(:,:,1) = sin(2*pi*dt*X).*cos(2*pi*dt*Y);
% psi_2(:,:,1) = sin(4*pi*dt*X).*cos(2*pi*dt*Y);
% psi_1(:,:,1) = -sin(1.2 * pi * X) .* sin(1.6 * pi * Y) + 0.6 * cos(2.4 * pi * X) .* cos(2.8 * pi * Y);
% psi_2(:,:,1) = sin(3.2 * pi * X) .* sin(0.8 * pi * Y) + 0.7 * cos(1.6 * pi * X) .* cos(2.4 * pi * Y);
psi_1(:,:,1) = exp(-(2 * (X - 1/2).^2 + (Y - 1/2).^2) / (2 * (1 / 8)^2));
psi_2(:,:,1) = exp(-((X - 1/2).^2 + 4 * (Y - 1/2).^2) / (3 * (1 / 8)^2));

% Initialize q for each layer
q_1 = zeros(N_x, N_y, N_t);
q_2 = zeros(N_x, N_y, N_t);

kernel = ones(7, 7) / 49;

% Initialize q for t=0 using equation (2)
q_1(:,:,1) = del2(psi_1(:,:,1), h_x, h_y) + B * Y + (kd2 / 2) * (psi_2(:,:,1) - psi_1(:,:,1));
q_2(:,:,1) = del2(psi_2(:,:,1), h_x, h_y) + B * Y + (kd2 / 2) * (psi_1(:,:,1) - psi_2(:,:,1));

q_1_smoothed = zeros(N_x,N_y,N_t);
q_2_smoothed = zeros(N_x,N_y,N_t);

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
    % [dq_1_dx, dq_1_dy] = central_finite_difference_gradient(q_1(:,:,m), h_x, h_y);

    % Layer 2
    [dpsi_2_dx, dpsi_2_dy] = gradient(psi_2(:,:,m), h_x, h_y);
    [dq_2_dx, dq_2_dy] = gradient(q_2(:,:,m), h_x, h_y);
    % [dq_2_dx, dq_2_dy] = central_finite_difference_gradient(q_2(:,:,m), h_x, h_y);

    % Update q using the provided method (Step 1) for each layer
    term_1_1 = dpsi_1_dy .* dq_1_dx;
    term_2_1 = dpsi_1_dx .* dq_1_dy;
    q_1(:,:, m+1) = q_1(:,:, m) + dt * (F + term_1_1 - term_2_1);
    q_1(1,:,m+1) = 0; q_1(:,1,m+1) = 0; q_1(end,:,m+1) = 0; q_1(:,end,m+1) = 0;

    % Layer 2
    term_1_2 = dpsi_2_dy .* dq_2_dx;
    term_2_2 = dpsi_2_dx .* dq_2_dy;
    q_2(:,:, m+1) = q_2(:,:, m) + dt * (F + term_1_2 - term_2_2);
    q_2(1,:,m+1) = 0; q_2(:,1,m+1) = 0; q_2(end,:,m+1) = 0; q_2(:,end,m+1) = 0;

    % Apply cubic smoothing spline using `csaps` to q_1 and q_2
    % q_1_smoothed(:,:,m+1) = fnval(csaps({x, y}, q_1(:, :, m + 1), 0.2), {x, y});
    % q_2_smoothed(:,:,m+1) = fnval(csaps({x, y}, q_2(:, :, m + 1), 0.2), {x, y});
    % Smooth q_1 using convolution (replicate borders to maintain dimensions)
    q_1_smoothed(:, :, m+1) = conv2(q_1(:, :, m+1), kernel, 'same');
    
    % Smooth q_2 similarly
    q_2_smoothed(:, :, m+1) = conv2(q_2(:, :, m+1), kernel, 'same');

    % For Step 2 (updating psi) by inverting the matrix equation for psi
    % Construct the right-hand side vector for layer 1 and layer 2
    idx1 = 1:N_x; idx2 = N_x+1: N_x: N_x*N_y; idx3 = N_x*N_y:-1:N_x*(N_y - 1) + 1; idx4 = N_x:N_x:N_x*N_y;
    C_1 = h_x^2 * reshape(q_1_smoothed(:,:,m+1) - B * Y - (kd2 / 2) * (psi_2(:,:,m)), [], 1);
    C_1(idx1) = 0; C_1(idx2) = 0; C_1(idx3) = 0; C_1(idx4) = 0;
    C_2 = h_x^2 * reshape(q_2_smoothed(:,:,m+1) - B * Y - (kd2 / 2) * (psi_1(:,:,m)), [], 1);
    C_2(idx1) = 0; C_2(idx2) = 0; C_2(idx3) = 0; C_2(idx4) = 0;
    % Solve for psi_1 and psi_2 at the current time step
    psi_1_vector = A_inv * C_1;
    psi_2_vector = A_inv * C_2;

    % Reshape the solution back into matrix form
    psi_1(:,:,m+1) = reshape(psi_1_vector, N_x, N_y);
    psi_2(:,:,m+1) = reshape(psi_2_vector, N_x, N_y);
end
toc

%%
% Setup video writer
outputDir = 'C:\Users\yje06\Desktop\ㅇㅎㄱ\ANU\2024_Semester_1\COMP3770\QGPV\video\'; % Need to change to the user's local dir
videoFileName = ['QGPV_combined_Simulation_Nt', num2str(N_t), 'T', num2str(T), '_Nx', num2str(N_x),'.mp4'];
fullVideoPath = [outputDir videoFileName];
v = VideoWriter(fullVideoPath, 'MPEG-4');
v.FrameRate = 5;  % Define the frame rate
skipfreq = 200;
open(v);  % Open the video file for writing

hFig = figure('WindowState', 'maximized');  % Create a new figure for video frames
for m = 1:N_t/skipfreq
    % Clear current figure
    figure(hFig);
    clf(hFig);

    % Plot q1
    subplot(2, 3, 1);
    imagesc(x, y, psi_1(:, :, skipfreq*m));
    axis equal tight;

    colorbar;
    title(sprintf('psi1 at t=%.4f', skipfreq*m * dt));

    subplot(2, 3, 2);
    imagesc(x, y, q_1(:, :, skipfreq*m));
    axis equal tight;

    colorbar;
    title(sprintf('q1 at t=%.4f', skipfreq*m * dt));

    subplot(2, 3, 3);
    imagesc(x, y, q_1_smoothed(:, :, skipfreq*m));
    axis equal tight;

    colorbar;
    title(sprintf('q1 smoothed at t=%.4f', skipfreq*m * dt));
    

    % Plot q2
    subplot(2, 3, 4);
    imagesc(x, y, psi_2(:, :, skipfreq*m));
    axis equal tight;

    colorbar;
    title(sprintf('psi2 at t=%.4f', skipfreq*m * dt));

    subplot(2, 3, 5);
    imagesc(x, y, q_2(:, :, skipfreq*m));
    axis equal tight;

    colorbar;
    title(sprintf('q2 at t=%.4f', skipfreq*m * dt));

    subplot(2, 3, 6);
    imagesc(x, y, q_2_smoothed(:, :, skipfreq*m));
    axis equal tight;

    colorbar;
    title(sprintf('q2 smoothed at t=%.4f', skipfreq*m * dt));
    drawnow;  % Update figure window

    % Capture the frame
    frame = getframe(hFig);
    writeVideo(v, frame);  % Write the frame to the video
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
