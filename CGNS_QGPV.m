clc; clear;
tic

% Initialize time-related parameters
T = 1;
N_t = 10000; % 10000
dt = T/N_t;
t = 0:dt:T;

% Initialize parameters for the mesh
x_min = 0; x_max = 1; y_min = 0; y_max = 1;
N_x = 50; N_y = N_x;  % Mesh size
h_x = (x_max-x_min)/(N_x-1);
h_y = (y_max-y_min)/(N_y-1);
x = linspace(x_min, x_max, N_x);
y = linspace(y_min, y_max, N_y);
[X, Y] = meshgrid(x, y);

% Parameters estimated based off typical literature values
% B = 1.0e-11;
% kd2 = 1.7e-9;
B = 1.0e-2;
kd2 = 1.0e1;
Omega = -(4 + h_x^2*kd2/2);

kernel = ones(7, 7) / 49;

% Initialize streamfunction for each layer at t=0 using stream_func
psi_1 = zeros(N_x, N_y, N_t); % Y of the layer 1 for CGNS
psi_2 = zeros(N_x, N_y, N_t); % Y of the layer 2 for CGNS
psi_1(:,:,1) = exp(-(2 * (X - 1/2).^2 + (Y - 1/2).^2) / (2 * (1 / 8)^2));
psi_2(:,:,1) = exp(-((X - 1/2).^2 + 4 * (Y - 1/2).^2) / (3 * (1 / 8)^2));

q_1_estimated = zeros(N_x, N_y, N_t);
q_2_estimated = zeros(N_x, N_y, N_t);

% Initialize q for each layer at t = 2 using equation (2)
% scaling corresponds to the magnitud of x_max - x_min?
q_1 = zeros(N_x, N_y, N_t);
q_2 = zeros(N_x, N_y, N_t);
q_1(:,:,1) = del2(psi_1(:,:,1), h_x, h_y) + B * Y + (kd2 / 2) * (psi_2(:,:,1) - psi_1(:,:,1)); % X of the layer 1 for CGNS
q_2(:,:,1) = del2(psi_2(:,:,1), h_x, h_y) + B * Y + (kd2 / 2) * (psi_1(:,:,1) - psi_2(:,:,1)); % X of the layer 2 for CGNS

q_1_estimated(:,:,1) = q_1(:,:,1);
% psi_1_estimated(:,:,1) = zeroBoundary(psi_1_estimated(:,:,1));
q_2_estimated(:,:,1) = q_2(:,:,1);
% psi_2_estimated(:,:,1) = zeroBoundary(psi_2_estimated(:,:,1));
q_1_smoothed = zeros(N_x,N_y,N_t);
q_2_smoothed = zeros(N_x,N_y,N_t);
q_1_estimated_smoothed = zeros(N_x,N_y,N_t);
q_2_estimated_smoothed = zeros(N_x,N_y,N_t);

% Should separate the noise parameters for each layer
R_f1 = 0.1 * eye(N_x); R_f1(1,1) = 0; % !!! R_f1 initial value !!! 
R_f2 = 0.1 * eye(N_x); R_f2(1,1) = 0; % !!! R_f2 initial value !!!

B_1 = sqrt(5); % !!! Stochastic influence on both layers of X !!!
b_2 = 1.4; % !!! Stochastic influence on both layers of Y !!!

% Construct A using the provided function
A = constructA(N_x, N_y, Omega);
A_inv = A\eye(size(A));
toc

%%
for m = 1:N_t - 1
    dw1 = sqrt(dt) * randn(N_x);
    dw2 = sqrt(dt) * randn(N_x);
    
    noise1 = B_1 * dw1;
    noise2 = 0;
    % noise2 = b_2 * dw2;

    % Update stream function derivatives for each layer
    % Layer 1
    [dpsi_1_dx, dpsi_1_dy] = gradient(psi_1(:,:,m), h_x, h_y);
    [dq_1_dx, dq_1_dy] = gradient(q_1(:,:,m), h_x, h_y);

    % Layer 2
    [dpsi_2_dx, dpsi_2_dy] = gradient(psi_2(:,:,m), h_x, h_y);
    [dq_2_dx, dq_2_dy] = gradient(q_2(:,:,m), h_x, h_y);

    % Update q using the provided method (Step 1) for each layer
    % Layer 1
    term_1_1 = dpsi_1_dy .* dq_1_dx;
    term_2_1 = dpsi_1_dx .* dq_1_dy;
    J_1 = term_2_1 - term_1_1;
    q_1(:,:, m+1) = q_1(:,:, m) + dt * (-J_1);
    
    % Layer 2
    term_1_2 = dpsi_2_dy .* dq_2_dx;
    term_2_2 = dpsi_2_dx .* dq_2_dy;
    J_2 = term_2_2 - term_1_2;
    q_2(:,:, m+1) = q_2(:,:, m) + dt * (-J_2);

    % Smooth q_1 using convolution (replicate borders to maintain dimensions)
    q_1_smoothed(:, :, m+1) = conv2(q_1(:, :, m+1), kernel, 'same');
    
    % Smooth q_2 similarly
    q_2_smoothed(:, :, m+1) = conv2(q_2(:, :, m+1), kernel, 'same');

    % For Step 2 (updating psi) by inverting the matrix equation for psi
    % Construct the right-hand side vector for layer 1 and layer 2
    idx1 = 1:N_x; idx2 = N_x+1: N_x: N_x*N_y; idx3 = N_x*N_y:-1:N_x*(N_y - 1) + 1; idx4 = N_x:N_x:N_x*N_y;
    C_1 = h_x^2 * reshape(q_1(:,:,m) - B * Y - (kd2 / 2) * (psi_2(:,:,m)), [], 1); % !!! m -> m+1 -> m
    C_1(idx1) = 0; C_1(idx2) = 0; C_1(idx3) = 0; C_1(idx4) = 0;
    C_2 = h_x^2 * reshape(q_2(:,:,m) - B * Y - (kd2 / 2) * (psi_1(:,:,m)), [], 1);
    C_2(idx1) = 0; C_2(idx2) = 0; C_2(idx3) = 0; C_2(idx4) = 0;
    
    % Solve for psi_1 and psi_2 at the current time step
    % Critical path even with the precomputation of A_inv as the size of A_inv is inherently large
    psi_1_vector = A_inv * C_1; 
    psi_2_vector = A_inv * C_2;
    
    % Reshape the solution back into matrix form
    psi_1(:,:,m+1) = reshape(psi_1_vector, N_x, N_y) + noise2;
    psi_2(:,:,m+1) = reshape(psi_2_vector, N_x, N_y) + noise2;

    % Apply CGNS to get the estimation
    % Compute the CGNS matrices
    A_0_1 = (((2/kd2)*del2(psi_1(:,:,m+1), h_x, h_y) + psi_2(:,:,m+1)) ...
        - ((2/kd2)*del2(psi_1(:,:,m), h_x, h_y) + psi_2(:,:,m))) / dt;
    A_0_2 = (((2/kd2)*del2(psi_2(:,:,m+1), h_x, h_y) + psi_1(:,:,m+1)) ...
        - ((2/kd2)*del2(psi_2(:,:,m), h_x, h_y) + psi_1(:,:,m))) / dt;

    a_0_1 = 0; a_0_2 = 0;
    
    [a_1_1, A_1_1] = construct_a1_A1(N_x,N_y,h_x,h_y,psi_1(:,:,m),kd2);
    [a_1_2, A_1_2] = construct_a1_A1(N_x,N_y,h_x,h_y,psi_2(:,:,m),kd2);
    
    % Adjust to the appropriate dimension
    q_1_vec = reshape(q_1_estimated(:,:,m), [], 1);
    q_2_vec = reshape(q_2_estimated(:,:,m), [], 1);

    A_1_q1 = A_1_1 * q_1_vec;
    a_1_q1 = a_1_1 * q_1_vec;
    A_1_q2 = A_1_2 * q_2_vec;
    a_1_q2 = a_1_2 * q_2_vec;

    % Reshape the results back to matrices
    A_1_q1 = reshape(A_1_q1, N_x, N_y);
    a_1_q1 = reshape(a_1_q1, N_x, N_y);
    A_1_q2 = reshape(A_1_q2, N_x, N_y);
    a_1_q2 = reshape(a_1_q2, N_x, N_y);
    
    % !second half of the residual term is wrong (A_0_1 -> del2(psi)), 
    % start estimating after some i iterations!
    psi_1_residual = reshape((psi_1(:,:,m+1) - psi_1(:,:,m)-(A_0_1 + A_1_q1)*dt), [], 1);
    dmu_f1_term = reshape(A_1_1' * psi_1_residual, N_x, N_y);
    dmu_f1 = (a_0_1 + a_1_q1) * dt + (1/B_1)^2*(R_f1*dmu_f1_term);

    % !Need to check this part!
    a_1_Rf1 = reshape(a_1_1*reshape(R_f1,[],1), N_x, N_y);
    R_f_a1 = reshape(a_1_1*reshape(R_f1',[],1), N_x,N_y)';
    A_1_Rf1 = reshape(A_1_1*reshape(R_f1,[],1), N_x, N_y);
    R_f_A1 = reshape(A_1_1*reshape(R_f1',[],1), N_x,N_y)';
    dR_f1 = (a_1_Rf1 + R_f_a1 + b_2^2 - (1/B_1^2)*R_f_A1*A_1_Rf1)*dt;

    R_f1 = R_f1 + dR_f1;
    if (m < 10)
        q_1_estimated(:,:,m+1) = q_1(:,:,m+1);
    else
        q_1_estimated(:,:,m+1) = q_1_estimated(:,:,m) + dmu_f1;
        % Smooth q_1_estimated using convolution (replicate borders to maintain dimensions)
        q_1_estimated_smoothed(:, :, m+1) = conv2(q_1_estimated(:, :, m+1), kernel, 'same');
    end

    psi_2_residual = reshape((psi_2(:,:,m+1) - psi_2(:,:,m)-(A_0_2 + A_1_q2)*dt), [], 1);
    dmu_f2_term = reshape(A_1_2' * psi_2_residual, N_x, N_y);
    dmu_f2 = (a_0_2 + a_1_q2) * dt + (1/B_1)^2*(R_f2*dmu_f2_term);

    a_1_Rf2 = reshape(a_1_2*reshape(R_f2,[],1), N_x, N_y);
    R_f_a2 = reshape(a_1_2*reshape(R_f2',[],1), N_x,N_y)';
    A_1_Rf2 = reshape(A_1_2*reshape(R_f2,[],1), N_x, N_y);
    R_f_A2 = reshape(A_1_2*reshape(R_f2',[],1), N_x,N_y)';
    dR_f2 = (a_1_Rf2 + R_f_a2 + b_2^2 - (1/B_1^2)*R_f_A2*A_1_Rf2)*dt;
    R_f2 = R_f2 + dR_f2;
    if (m < 10)
        q_2_estimated(:,:,m+1) = q_2(:,:,m+1);
    else
        q_2_estimated(:,:,m+1) = q_2_estimated(:,:,m) + dmu_f2;
        % Smooth q_2_estimated similarly
        q_2_estimated_smoothed(:, :, m+1) = conv2(q_2_estimated(:, :, m+1), kernel, 'same');
    end
    % check symmetry of a_1 and R_f1
    % dmu_f = (a_0 + a_1*q_f) * dt + (1/B_1)^2*(R_f*A_1')*(dPsi - (A_0 + A_1*q_f)*dt);
    % dR_f = (a_1*R_f + R_f*a_1' + b_2^2 - (1/B_1^2)*(R_f*A_1')*(A_1*R_f))*dt;
end
toc

%% Visualization
% Difference between the true and estimated to see the performance of CGNS
q_1_diff_smoothed = (q_1_smoothed - q_1_estimated_smoothed);
q_2_diff_smoothed = (q_2_smoothed - q_2_estimated_smoothed);

% Setup video writer
outputDir = 'C:\Users\yje06\Desktop\ㅇㅎㄱ\ANU\2024_Semester_1\COMP3770\Artefact\videos\'; % Need to change to the user's local dir
videoFileName = ['QGPV_CGNS_smoothed_Simulation_Nt', num2str(N_t), 'T', num2str(T), '_Nx', num2str(N_x),'.mp4'];
fullVideoPath = [outputDir videoFileName];
v = VideoWriter(fullVideoPath, 'MPEG-4');
v.FrameRate = 5;  % Define the frame rate 150
skipfreq = 200;
open(v);  % Open the video file for writing

Fig = figure('WindowState', 'maximized');  % Create a new figure for video
for m = 1:N_t/skipfreq
    % Clear current figure
    figure(Fig);
    clf(Fig);
    
    % layer 1
    subplot(2,3,1);
    imagesc(x, y, q_1_smoothed(:, :, skipfreq * m));
    axis equal tight;
    colorbar;
    title(sprintf('q1 true at t=%.4f', skipfreq *  m * dt));
    
    subplot(2,3,2);
    imagesc(x, y, q_1_estimated_smoothed(:, :, skipfreq * m));
    axis equal tight;
    colorbar;
    title(sprintf('q1 estimated at t=%.4f', skipfreq * m * dt));

    subplot(2,3,3);
    imagesc(x, y, q_1_diff_smoothed(:, :, skipfreq * m));
    axis equal tight;
    colorbar;
    title(sprintf('q1 difference at t=%.4f', skipfreq * m * dt));
    
    % layer 2
    subplot(2,3,4);
    imagesc(x, y, q_2_smoothed(:, :, skipfreq * m));
    axis equal tight;
    colorbar;
    title(sprintf('q2 true at t=%.4f', skipfreq * m * dt));
   
    subplot(2,3,5);
    imagesc(x, y, q_2_estimated_smoothed(:, :, skipfreq * m));
    axis equal tight;
    colorbar;
    title(sprintf('q2 estimated at t=%.4f', skipfreq * m * dt));

    subplot(2,3,6);
    imagesc(x, y, q_2_diff_smoothed(:, :, skipfreq * m));
    axis equal tight;
    colorbar;
    title(sprintf('q2 difference at t=%.4f', skipfreq * m * dt));

    % Capture the frame and draw
    drawnow;
    frame = getframe(Fig);
    writeVideo(v, frame);
end
close(v);
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
    
    idx1 = 1:N_x;
    idx2 = N_x+1: N_x: N_x*N_y;
    idx3 = N_x*N_y:-1:N_x*(N_y - 1) + 1;
    idx4 = N_x:N_x:N_x*N_y;

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

%%
function [a1,A1] = construct_a1_A1(N_x,N_y,h_x,h_y,psi,kd2)
    % Total number of grid points
    N = N_x * N_y;
    
    % Initialize A with zeros
    a1 = zeros(N, N);
    
    % Helper function to convert 2D indices to 1D index
    idx = @(x, y) (mod(x-1, N_x)) * N_y + mod(y-1, N_y) + 1;
    % psi_test = zeros(N_x, N_y);
    % psi_test(1,:) = [1,5,9,13];
    % psi_test(2,:) = [2,6,10,14];
    % psi_test(3,:) = [3,7,11,15];
    % psi_test(4,:) = [4,8,12,16];
    for x = 1:N_x
        for y = 1:N_y
            % Current position index
            currentIdx = idx(x, y);
           
            % Adjacent indices (accounting for periodic boundary conditions)
            idxLeft = idx(x-1, y);
            idxRight = idx(x+1, y);
            idxUp = idx(x, y-1);
            idxDown = idx(x, y+1);
            
            idxLeftPsi= mod(x-2, N_x) + 1;
            idxRightPsi = mod(x, N_x) + 1;
            idxDownPsi = mod(y, N_y) + 1;
            idxUpPsi = mod(y-2, N_y) + 1;
            a1(currentIdx, idxLeft) = psi(idxDownPsi, x) - psi(idxUpPsi, x);
            a1(currentIdx, idxRight) = psi(idxUpPsi, x) - psi(idxDownPsi, x);
            a1(currentIdx, idxUp) = psi(y, idxLeftPsi) - psi(y, idxRightPsi);
            a1(currentIdx, idxDown) = psi(y, idxRightPsi) - psi(y, idxLeftPsi);
        end
    end
    
    scaling_factor = 1; % is this valid?
    a1 = scaling_factor * (-1/(4*h_x*h_y)) * a1;

    idx1 = 1:N_x;
    idx2 = N_x+1: N_x: N_x*N_y;
    idx3 = N_x*N_y:-1:N_x*(N_y - 1) + 1;
    idx4 = N_x:N_x:N_x*N_y;

    A(idx1, :) = 0; 
    for j = 1:length(idx1)
        A(idx1(j),idx1(j)) = 1;
    end
   
    A(idx2, :) = 0;
    for j = 1:length(idx2)
        A(idx2(j),idx2(j)) = 1;
    end

    a1(idx3, :) = 0;
    for j = 1:length(idx3)
        A(idx3(j),idx3(j)) = 1;
    end

    a1(idx4, :) = 0;
    for j = 1:length(idx4)
        A(idx4(j),idx4(j)) = 1;
    end


    A1 = (-2/kd2) * a1;
    

end