    clc; clear;
    tic
    
    % Initialize time-related parameters
    T = 0.1;
    N_t = 1000;
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
    
    smooth_size = 4;
    kernel = ones(smooth_size, smooth_size) / (smooth_size^2);
    
    % Initialize streamfunction for each layer at t=0 using stream_func
    psi_1 = zeros(N_x, N_y, N_t); % Y of the layer 1 for CGNS
    psi_2 = zeros(N_x, N_y, N_t); % Y of the layer 2 for CGNS
    psi_2(:,:,1) = -sin(1.2 * pi * X) .* sin(1.5 * pi * Y) + 0.6 * cos(2.3 * pi * X) .* cos(2.8 * pi * Y);
    psi_1(:,:,1) = sin(3.1 * pi * X) .* sin(0.8 * pi * Y) + 0.7 * cos(1.6 * pi * X) .* cos(2.4 * pi * Y);
    % psi_1(:,:,1) = exp(-(2 * (X - 1/2).^2 + (Y - 1/2).^2) / (2 * (1 / 8)^2));
    % psi_2(:,:,1) = exp(-((X - 1/2).^2 + 4 * (Y - 1/2).^2) / (3 * (1 / 8)^2));
    
    psi_2_estimated = zeros(N_x, N_y, N_t);
    
    % Initialize q for each layer at t = 2 using equation (2)
    % scaling corresponds to the magnitud of x_max - x_min?
    q_1 = zeros(N_x, N_y, N_t);
    q_2 = zeros(N_x, N_y, N_t);
    q_1(:,:,1) = del2(psi_1(:,:,1), h_x, h_y) + B * Y + (kd2 / 2) * (psi_2(:,:,1) - psi_1(:,:,1)); % X of the layer 1 for CGNS
    q_2(:,:,1) = del2(psi_2(:,:,1), h_x, h_y) + B * Y + (kd2 / 2) * (psi_1(:,:,1) - psi_2(:,:,1)); % X of the layer 2 for CGNS
    
    q_1(1,:,1) = 0; q_1(:,1,1) = 0;
    q_1(end,:,1) = 0; q_1(:,end,1) = 0;
    q_2(1,:,1) = 0; q_2(:,1,1) = 0;
    q_2(end,:,1) = 0; q_2(:,end,1) = 0;
    
    % Should separate the noise parameters for each layer
    B_1 = sqrt(1); % !!! Stochastic influence on both layers of X !!!
    b_2 = 0.1; % !!! Stochastic influence on both layers of Y !!!

    % Construct A using the provided function
    A = constructA(N_x, N_y, Omega);
    A_inv = A\eye(size(A));
    toc
    
    %%
    for m = 1:(N_t/100) - 1
        dw1 = sqrt(dt) * rand(N_x, N_y);
        dw2 = sqrt(dt) * rand(N_x, N_y);
        
        noise1 = B_1 * dw1;
        noise2 = b_2 * dw2;
        % noise1 = 0;
        % noise2 = 0;
    
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
        q_1(:,:, m+1) = q_1(:,:, m) + dt * (-J_1) + noise2;
        
        % Layer 2
        term_1_2 = dpsi_2_dy .* dq_2_dx;
        term_2_2 = dpsi_2_dx .* dq_2_dy;
        J_2 = term_2_2 - term_1_2;
        q_2(:,:, m+1) = q_2(:,:, m) + dt * (-J_2) + noise2;
    
        q_1(1,:,m+1) = 0; q_1(:,1,m+1) = 0;
        q_1(end,:,m+1) = 0; q_1(:,end,m+1) = 0;
    
        q_2(1,:,m+1) = 0; q_2(:,1,m+1) = 0;
        q_2(end,:,m+1) = 0; q_2(:,end,m+1) = 0;
    
        % For Step 2 (updating psi) by inverting the matrix equation for psi
        % Construct the right-hand side vector for layer 1 and layer 2
        idx1 = 1:N_x; idx2 = N_x+1: N_x: N_x*N_y; idx3 = N_x*N_y:-1:N_x*(N_y - 1) + 1; idx4 = N_x:N_x:N_x*N_y;
        C_1 = h_x^2 * reshape(q_1(:,:,m + 1) - B * Y - (kd2 / 2) * (psi_2(:,:,m)), [], 1); % !!! m -> m+1 -> m
        C_1(idx1) = 0; C_1(idx2) = 0; C_1(idx3) = 0; C_1(idx4) = 0;
        C_2 = h_x^2 * reshape(q_2(:,:,m + 1) - B * Y - (kd2 / 2) * (psi_1(:,:,m)), [], 1);
        C_2(idx1) = 0; C_2(idx2) = 0; C_2(idx3) = 0; C_2(idx4) = 0;
        
        % Solve for psi_1 and psi_2 at the current time step
        psi_1_vector = A_inv * C_1; 
        psi_2_vector = A_inv * C_2;
        
        % Reshape the solution back into matrix form
        psi_1(:,:,m+1) = reshape(psi_1_vector, N_x, N_y) + noise1;
        psi_2(:,:,m+1) = reshape(psi_2_vector, N_x, N_y) + noise1;
        psi_2_estimated(:,:,m+1) = psi_2(:,:,m+1);
    end
    toc
    
    psi_1_sum_cov = zeros(N_x, N_y);
    psi_2_sum_cov = zeros(N_x, N_y);
    for m = 1:N_t/100
        psi_1_sum_cov = psi_1_sum_cov + psi_1(:,:,m);
        psi_2_sum_cov = psi_2_sum_cov + psi_2(:,:,m);
    end
    
    psi_1_mean_cov = (100/N_t) * psi_1_sum_cov;
    psi_2_mean_cov = (100/N_t) * psi_2_sum_cov;

    Cov_xx = zeros(N_x, N_y);
    Cov_xy = zeros(N_x, N_y);
    Cov_yy = zeros(N_y, N_y);

    
    for m = 1:N_t/100
        Cov_xx = Cov_xx + (psi_1(:,:,m) - psi_1_mean_cov).^2;
        Cov_xy = Cov_xy + (psi_2(:,:,m) - psi_2_mean_cov) .* (psi_1(:,:,m) - psi_1_mean_cov);
        Cov_yy = Cov_yy + (psi_2(:,:,m) - psi_2_mean_cov).^2;
    end
    Cov_xx = (100/N_t) * Cov_xx;
    Cov_xy = (100/N_t) * Cov_xy;
    Cov_yy = (100/N_t) * Cov_yy;

    R_f1 = Cov_yy - (Cov_xy ./ Cov_xx) .* Cov_xy; 
    toc
    
    RMSE_1 = zeros(N_x, N_y);
    mean_1 = zeros(N_x, N_y);
    mean_estimated_1 = zeros(N_x, N_y);
    
    for m = N_t/100:N_t - 1
        dw1 = sqrt(dt) * rand(N_x,N_y);
        dw2 = sqrt(dt) * rand(N_x,N_y);
        
        noise1 = B_1 * dw1;
        noise2 = b_2 * dw2;
    
        % Update stream function derivatives for each layer
        q_1(:,:, m+1) = q_1(:,:, m) + dt * (-Jacobian(psi_1,q_1,m,h_x,h_y));
        
        % Layer 2
        q_2(:,:, m+1) = q_2(:,:, m) + dt * (-Jacobian(psi_2,q_2,m,h_x,h_y));
    
        q_1(1,:,m+1) = 0; q_1(:,1,m+1) = 0;
        q_1(end,:,m+1) = 0; q_1(:,end,m+1) = 0;
    
        q_2(1,:,m+1) = 0; q_2(:,1,m+1) = 0;
        q_2(end,:,m+1) = 0; q_2(:,end,m+1) = 0;
    
        % For Step 2 (updating psi) by inverting the matrix equation for psi
        % Construct the right-hand side vector for layer 1 and layer 2
        idx1 = 1:N_x; idx2 = N_x+1: N_x: N_x*N_y; idx3 = N_x*N_y:-1:N_x*(N_y - 1) + 1; idx4 = N_x:N_x:N_x*N_y;
        C_1 = h_x^2 * reshape(q_1(:,:,m + 1) - B * Y - (kd2 / 2) * (psi_2(:,:,m)), [], 1); % !!! m -> m+1 -> m
        C_1(idx1) = 0; C_1(idx2) = 0; C_1(idx3) = 0; C_1(idx4) = 0;
        C_2 = h_x^2 * reshape(q_2(:,:,m + 1) - B * Y - (kd2 / 2) * (psi_1(:,:,m)), [], 1);
        C_2(idx1) = 0; C_2(idx2) = 0; C_2(idx3) = 0; C_2(idx4) = 0;
        
        % Solve for psi_1 and psi_2 at the current time step
        psi_1_vector = A_inv * C_1; 
        psi_2_vector = A_inv * C_2;
        
        % Reshape the solution back into matrix form
        psi_1(:,:,m+1) = reshape(psi_1_vector, N_x, N_y) + noise1;
        psi_2(:,:,m+1) = reshape(psi_2_vector, N_x, N_y) + noise2;
        
        % TEST
        % psi_diff = -2/kd2 * Jacobian(psi_1,q_1,m,h_x,h_y) + (((psi_1(:,:,m) - (2/kd2) * del2(psi_1(:,:,m), h_x, h_y)) - (psi_1(:,:,m-1) - (2/kd2) * del2(psi_1(:,:,m-1), h_x, h_y))) / dt);
    
    
        % Apply CGNS to get the estimation
        % Compute the CGNS matrices
        % Apply CGNS to get the estimation
        % Compute the CGNS matrices
        % A_0 = (((-2/kd2)*del2(psi_2(:,:,m), h_x, h_y) + psi_2(:,:,m)) ...
        %       - ((-2/kd2)*del2(psi_2(:,:,m-1), h_x, h_y) + psi_2(:,:,m-1))) /dt;
        % a_0 = (((-2/kd2)*del2(psi_1(:,:,m), h_x, h_y) + psi_1(:,:,m)) ...
        %       - ((-2/kd2)*del2(psi_1(:,:,m-1), h_x, h_y) + psi_1(:,:,m-1))) / dt;
        % 
        % A_1 = (-2/kd2) * construct_A1(N_x,N_y,h_x,h_y,q_2(:,:,m));
        % [a_0_part, a_1] = construct_a1(N_x,N_y,h_x,h_y,psi_1(:,:,m),kd2,B);
        % 
        % a_0 = a_0 + a_0_part;
    
        % dmu_f = (a_0 + a_1*q_f) * dt + (1/B_1)^2*(R_f*A_1')*(dX - (A_0 + A_1*q_f)*dt);
        % dR_f = (a_1*R_f + R_f*a_1' + b_2^2 - (1/B_1^2)*(R_f*A_1')*(A_1*R_f))*dt;
        A_0 = (1/dt) * (h_x^2 * A_inv * (reshape(q_1(:,:,m+1) - q_1(:,:,m) + (kd2/2) * psi_2_estimated(:,:,m-1), [], 1))); A_0 = reshape(A_0, N_x, N_y);
        A_1 = (1/dt) * ((-h_x^2*kd2/2) * A_inv);
        a_0 = (1/dt) * (h_x^2*A_inv*(reshape(q_2(:,:,m) - q_2(:,:,m-1) -(kd2/2) * (psi_1(:,:,m-1) - psi_1(:,:,m-2)), [], 1))); a_0 = reshape(a_0, N_x, N_y);
        a_1 = 0;

        psi_2_vec = reshape(psi_2_estimated(:,:,m), [], 1);
    
        A_1_psi2 = A_1 * psi_2_vec;
        a_1_psi2 = a_1 * psi_2_vec;
    
        % Reshape the results back to matrices
        A_1_psi2 = reshape(A_1_psi2, N_x, N_y);
        a_1_psi2 = reshape(a_1_psi2, N_x, N_y);                                                  
        
        psi_1_residual = reshape((psi_1(:,:,m+1) - psi_1(:,:,m)-(A_0 + A_1_psi2)*dt), [], 1);
        dmu_f1_term = reshape(A_1' * psi_1_residual, N_x, N_y);
        dmu_f1 = (a_0 + a_1_psi2) * dt + (1/B_1)^2*(R_f1.*dmu_f1_term);
    
        a_1_Rf1 = reshape(a_1*reshape(R_f1,[],1), N_x, N_y);
        R_f_a1 = reshape(a_1*reshape(R_f1',[],1), N_x,N_y)';
        dR_f1 = (a_1_Rf1 + R_f_a1 + b_2^2 - (1/B_1^2)*R_f1 .* reshape((A_1.^2 * reshape(R_f1,[],1)), N_x, N_y))*dt;

        R_f1 = R_f1 + dR_f1;
        psi_2_estimated(:,:,m+1) = psi_2_estimated(:,:,m) + dmu_f1;

        RMSE_1 = RMSE_1 + (psi_2_estimated(:,:,m+1) - psi_2(:,:,m+1)).^2;
        mean_1 = mean_1 + psi_2(:,:,m+1);
        mean_estimated_1 = mean_estimated_1 + psi_2_estimated(:,:,m+1);

    end
    toc
    
    %% Visualization
    mean_1 = mean_1 / (99*N_t/100);
    mean_estimated_1 = mean_estimated_1 / (99*N_t/100);
    std_1 = zeros(N_x,N_y);
    std_estimated_1 = zeros(N_x,N_y);
    corr_combined_1 = zeros(N_x,N_y);
    %%
    for m = (N_t/100) + 1:N_t
        true_diff_1 = (psi_2(:,:,m) - mean_1);
        estimated_diff_1 = (psi_2_estimated(:,:,m) - mean_estimated_1);
        std_estimated_1 = std_estimated_1 + estimated_diff_1.^2;
        std_1 = std_1 + true_diff_1.^2;
        corr_combined_1 = corr_combined_1 + estimated_diff_1 .* true_diff_1;
    end
    std_corr_1 = std_1;
    std_1 = sqrt(std_1 / (99*N_t/100));
    RMSE_1 = sqrt(RMSE_1 / (99*N_t/100));
    corr_1 = corr_combined_1 ./ sqrt(std_estimated_1 .* std_corr_1);
    
    RMSE = mean(RMSE_1(:));
    corr = mean(corr_1(:));
    fprintf('RMSE: %.4f, corr: %.4f', RMSE, corr);

    figure; 
    subplot(1,2,1); imagesc(RMSE_1); axis equal tight; colorbar; 
    subplot(1,2,2); imagesc(corr_1); axis equal tight; colorbar;
    %%
    % Difference between the true and estimated to see the performance of CGNS
    psi_2_diff = (psi_2 - psi_2_estimated);
    
    % Setup video writer
    outputDir = 'C:\Users\yje06\Desktop\ㅇㅎㄱ\ANU\2024_Semester_1\COMP3770\Artefact\videos\psi_final\'; % Need to change to the user's local dir
    videoFileName = ['prop_cov_noise_exp_CGNS_psi',num2str(smooth_size),'_Simulation_Nt', num2str(N_t), 'T', num2str(T), '_Nx', num2str(N_x),'.mp4'];
    fullVideoPath = [outputDir videoFileName];
    v = VideoWriter(fullVideoPath, 'MPEG-4');
    v.FrameRate = 15;  % Define the frame rate 150
    skipfreq = 100;
    open(v);  % Open the video file for writing
    
    Fig = figure('WindowState', 'maximized');  % Create a new figure for video
    for m = 1:N_t/skipfreq
        % Clear current figure
        figure(Fig);
        clf(Fig);
        
        % layer 1
        subplot(1,3,1);
        imagesc(x, y, psi_2(:, :, skipfreq * m));
        axis equal tight;
        colorbar;
        title(sprintf('psi2 true at t=%.4f', skipfreq *  m * dt));
        
        subplot(1,3,2);
        imagesc(x, y, psi_2_estimated(:, :, skipfreq * m));
        axis equal tight;
        colorbar;
        title(sprintf('psi2 estimated at t=%.4f', skipfreq * m * dt));
    
        subplot(1,3,3);
        imagesc(x, y, psi_2_diff(:, :, skipfreq * m));
        axis equal tight;
        colorbar;
        title(sprintf('psi2 difference at t=%.4f', skipfreq * m * dt));
    
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
    
    % %%
    % function [A1] = construct_A1(N_x,N_y,h_x,h_y,q)
    %     % Total number of grid points
    %     N = N_x * N_y;
    % 
    %     % Initialize A with zeros
    %     A1 = zeros(N, N);
    % 
    %     % Helper function to convert 2D indices to 1D index
    %     idx = @(x, y) (mod(x-1, N_x)) * N_y + mod(y-1, N_y) + 1;
    %     for x = 1:N_x
    %         for y = 1:N_y
    %             % Current position index
    %             currentIdx = idx(x, y);
    % 
    %             % Adjacent indices (accounting for periodic boundary conditions)
    %             idxLeft = idx(x-1, y);
    %             idxRight = idx(x+1, y);
    %             idxUp = idx(x, y-1);
    %             idxDown = idx(x, y+1);
    % 
    %             idxLeftQ= mod(x-2, N_x) + 1;
    %             idxRightQ = mod(x, N_x) + 1;
    %             idxDownQ = mod(y, N_y) + 1;
    %             idxUpQ = mod(y-2, N_y) + 1;
    %             A1(currentIdx, idxLeft) = q(idxDownQ, x) - q(idxUpQ, x);
    %             A1(currentIdx, idxRight) = q(idxUpQ, x) - q(idxDownQ, x);
    %             A1(currentIdx, idxUp) = q(y, idxLeftQ) - q(y, idxRightQ);
    %             A1(currentIdx, idxDown) = q(y, idxRightQ) - q(y, idxLeftQ);
    %         end
    %     end
    %     A1 = (-1/(4*h_x*h_y)) * A1;
    % 
    %     idx1 = 1:N_x;
    %     idx2 = N_x+1: N_x: N_x*N_y;
    %     idx3 = N_x*N_y:-1:N_x*(N_y - 1) + 1;
    %     idx4 = N_x:N_x:N_x*N_y;
    % 
    %     A1(idx1, :) = 0; 
    %     A1(idx2, :) = 0;
    %     A1(idx3, :) = 0;
    %     A1(idx4, :) = 0;
    % end
    % 
    % %%
    % function [a_0_part, a1] = construct_a1(N_x,N_y,h_x,h_y,psi_1, kd2, B)
    %     % Total number of grid points
    %     N = N_x * N_y;
    % 
    %     % Initialize A with zeros
    %     a1 = zeros(N, N);
    %     a_0_part = zeros(N_x,N_y);
    %     % Helper function to convert 2D indices to 1D index
    %     idx = @(x, y) (mod(x-1, N_x)) * N_y + mod(y-1, N_y) + 1;
    %     for x = 1:N_x
    %         for y = 1:N_y
    %             % Current position index
    %             currentIdx = idx(x, y);
    % 
    %             % Adjacent indices (accounting for periodic boundary conditions)
    %             idxLeft = idx(x-1, y); 
    %             idxRight = idx(x+1, y);
    %             idxUp = idx(x, y-1); 
    %             idxDown = idx(x, y+1); 
    % 
    %             idxLeftPsi= mod(x-2, N_x) + 1; idxLeftLeftPsi = mod(x-3, N_x) + 1;
    %             idxRightPsi = mod(x, N_x) + 1; idxRightRightPsi = mod(x+1, N_x) + 1;
    %             idxDownPsi = mod(y, N_y) + 1; idxDownDownPsi = mod(y+1,N_y) + 1;
    %             idxUpPsi = mod(y-2, N_y) + 1; idxUpUpPsi = mod(y-3,N_y) + 1;
    % 
    %             const = kd2/(8*h_x*h_y);
    %             curr1 = (psi_1(y, idxRightPsi) - psi_1(y, idxLeftPsi)) .* ...
    %                     ((psi_1(idxUpUpPsi, x) + psi_1(idxUpPsi,idxRightPsi) + psi_1(idxUpPsi,idxLeftPsi)-4*psi_1(idxUpPsi,x) ...
    %                     - psi_1(idxDownDownPsi,x) - psi_1(idxDownPsi,idxRightPsi) - psi_1(idxDownPsi,idxLeftPsi) + 4*psi_1(idxDownPsi,x))/(h_x*h_y) + ...
    %                     2*B*h_y + (kd2/2) * (-psi_1(idxUpPsi,x) + psi_1(idxDownPsi,x)));
    %             curr2 = (psi_1(idxUpPsi,x) - psi_1(idxDownPsi, x)) .* ...
    %                     ((psi_1(idxUpPsi, idxRightPsi) + psi_1(idxDownPsi,idxRightPsi) + psi_1(y,idxRightRightPsi)-4*psi_1(y,idxRightPsi) ...
    %                     - psi_1(idxUpPsi,idxLeftPsi) - psi_1(idxDownPsi,idxLeftPsi) - psi_1(y,idxLeftLeftPsi) + 4*psi_1(y,idxLeftPsi))/(h_x*h_y) + ...
    %                     (kd2/2) * (-psi_1(y,idxRightPsi) + psi_1(y,idxLeftPsi)));
    % 
    %             a_0_part(y,x) = (curr1 - curr2)./(4*h_x*h_y); % check psi_2 index
    % 
    %             a1(currentIdx, idxLeft) = const .* (psi_1(idxUpPsi, x) - psi_1(idxDownPsi, x));
    %             a1(currentIdx, idxRight) = const .* (psi_1(idxDownPsi, x) - psi_1(idxUpPsi, x));
    %             a1(currentIdx, idxUp) = const .* (psi_1(y, idxRightPsi) - psi_1(y, idxLeftPsi));
    %             a1(currentIdx, idxDown) = const .* (psi_1(y, idxLeftPsi) - psi_1(y, idxRightPsi));
    %         end
    %     end
    % 
    %     idx1 = 1:N_x;
    %     idx2 = N_x+1: N_x: N_x*N_y;
    %     idx3 = N_x*N_y:-1:N_x*(N_y - 1) + 1;
    %     idx4 = N_x:N_x:N_x*N_y;
    % 
    %     a1(idx1, :) = 0; 
    %     a1(idx2, :) = 0;
    %     a1(idx3, :) = 0;
    %     a1(idx4, :) = 0;
    % 
    %     a_0_part = (-2/kd2) * a_0_part;
    %     a1 = (-2/kd2) * a1;
    % end
    
    %%
    function J = Jacobian(psi, q, m, h_x, h_y)
        % [N_x, N_y] = size(psi(:,:,m));
        [dpsi_1_dx, dpsi_1_dy] = gradient(psi(:,:,m), h_x, h_y);
        [dq_1_dx, dq_1_dy] = gradient(q(:,:,m), h_x, h_y);
    
        % Update q using the provided method (Step 1) for each layer
        % Layer 1
        term_1_1 = dpsi_1_dy .* dq_1_dx;
        term_2_1 = dpsi_1_dx .* dq_1_dy;
        J = term_2_1 - term_1_1;
    end
