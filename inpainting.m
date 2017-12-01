clear;
close all;
img = imread('test-img.tif');
mask = imread('test-mask.tif');
imshow(img);
figure;
imshow(mask);
masked_img = img + mask;
figure;
imshow(masked_img);
masked_img = anisodiff2D(masked_img, 1, 1/7, 30, 1);
figure;
imshow(masked_img,[]);
original_masked_img = masked_img;
figure;
for i=1:100
    display(i);
    for j=1:15
        masked_img = inpaint(masked_img, mask, 0.001);
    end
    for j=1:2
        diffused_img = anisodiff2D(masked_img, 5, 1/7, 30, 1);
        diffused_img(mask == 0) = 0;
        masked_img = original_masked_img;
        masked_img(mask ~= 0) = 0;
        masked_img = masked_img + diffused_img;
    end
    if mod(i, 10) == 0
        figure;
        imshow(uint8(masked_img));    
    end
end
masked_img = uint8(masked_img);
figure;
imshow(masked_img);

function inpainted_img = inpaint(img, mask, delta_t)
    update = get_update(img, mask);
    inpainted_img = img + delta_t * update;
end

function img_t = get_update(img, mask)
    beta = get_beta(img);
    mag_grad = get_gradient_magnitude(img, beta);
    img_t = beta.*mag_grad;
    img_t(mask == 0) = 0;
    threshold = 600;
    img_t(img_t > threshold) = threshold;
    img_t(img_t < -threshold) = -threshold;
end

function [l_i_diff, l_j_diff] = get_delta_l(img)
    l = get_l(img);
    l_i_diff = l(3:end,:) - l(1:end-2,:);
    l_j_diff = l(:, 3:end) - l(:, 1:end-2);
    l_i_diff = padarray(l_i_diff, [1 0]);
    l_j_diff = padarray(l_j_diff, [0 1]);
end

function l = get_l(img)
    filt = fspecial('laplacian', 0);
    l = conv2(img, filt);
    l = l(2:end-1, 2:end-1);
end

function [n_norm_x, n_norm_y] = get_norm_n(img)
    x_diff = img(3:end, :) - img(1:end-2, :);
    y_diff = img(:, 3:end) - img(:, 1:end-2);
    x_diff = padarray(x_diff, [1 0]);
    y_diff = padarray(y_diff, [0 1]);
    mag = sqrt(x_diff.^2 + y_diff.^2);
    n_norm_x = x_diff ./ mag;
    n_norm_y = -y_diff ./ mag;
    n_norm_x(mag == 0) = 0;
    n_norm_y(mag == 0) = 0;
        
end

function beta = get_beta(img)
    [delt_l_x, delt_l_y] = get_delta_l(img);
    [n_norm_x, n_norm_y] = get_norm_n(img);
    beta = delt_l_x.*n_norm_x + delt_l_y.*n_norm_y;
end

function mag_grad = get_gradient_magnitude(img, beta)
    img_diff_x = img(2:end,:)-img(1:end-1,:);
    img_diff_x_forward = padarray(img_diff_x, [1 0], 'replicate', 'post');
    img_diff_x_backward = padarray(img_diff_x, [1 0], 'replicate', 'pre');
    img_diff_y = img(:, 2:end) - img(:, 1:end-1);
    img_diff_y_forward = padarray(img_diff_y, [0 1], 'replicate', 'post');
    img_diff_y_backward = padarray(img_diff_y, [0 1], 'replicate', 'pre');
    
    img_diff_x_forward(beta >= 0 & img_diff_x_forward < 0) = 0;
    img_diff_x_backward(beta >= 0 & img_diff_x_backward > 0) = 0;
    img_diff_y_forward(beta >= 0 & img_diff_y_forward < 0) = 0;
    img_diff_y_backward(beta >= 0 & img_diff_y_backward > 0) = 0;
    img_diff_x_forward(beta < 0 & img_diff_x_forward > 0) = 0;
    img_diff_x_backward(beta < 0 & img_diff_x_backward < 0) = 0;
    img_diff_y_forward(beta < 0 & img_diff_y_forward > 0) = 0;
    img_diff_y_backward(beta < 0 & img_diff_y_backward < 0) = 0;
    
    mag_grad = sqrt(img_diff_x_forward.^2 + img_diff_x_backward.^2+img_diff_y_forward.^2+img_diff_y_backward.^2);
    %threshold = 100;
    %mag_grad(mag_grad > threshold) = threshold;
end