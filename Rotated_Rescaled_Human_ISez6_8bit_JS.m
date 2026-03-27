% % % ISez shape for human OCT image
% 2022-10-03, Haohua Qian
% 2022-10-05, add plot of outer retina intensity
% 2022-10-30, put cursor on start and stop and output is centered
% 2024-09-14, added rescaling - James Soetedjo.
% 2024-09-27 Set y-axis to be between -25 to 120 James Soetedjo.
[oct_file, oct_path]=uigetfile ('*.tif','Select OCT image file');  % oct image file
oct=imread(fullfile(oct_path, oct_file));

% noct=uint8(oct*255/max(oct(:)));  % normalize to maximum
noct=oct;
oct=oct/3;   % scale intensity to one third
% 
% sini=52;   % initial IS band starting point
% eini=84;   % initial IS band ending point

a_limit = 1;

for i=1:21
    if i>20                % fovea
        left=95;
        right=105;
        low=300;
        suffix='fovea';
    else
        left=2851-120*i;    % ROI left side
        right=2850-120*(i-1);
        low=300;
        suffix=sprintf('%02d', i);
    end
    
    roct=noct;
    roct(450,left:right)=255;          % ROI
    roct(low,left:right)=255;
    roct(low:450,left)=255;
    roct(low:450,right)=255;

    imwrite(roct,fullfile(oct_path, [oct_file(1:end-4) '_ROI_' suffix '.jpg']));

    Ioct=oct(low:450,left:right);

    Int=mean(Ioct,2);  % average intensity profile 

    f=figure(2);
%     ymin=0;
    ymin=min(Int);
    f.Position = [50 300 1085 328];
    plot(Int)
    set(gca,'xlim',[0 140])
    set(gca,'Ylim',[ymin ymin+80])
    
    [x_left,~, ~] = ginput(1);
    hold on
    plot([x_left x_left],[ymin ymin+140])
    s=round(x_left);

    [x_right,~, ~] = ginput(1);
    plot([x_right x_right],[ymin ymin+140])
    e=round(x_right);
    center=(s+e)/2;    % center position of IS band
    hold off

    saveas(gcf,fullfile(oct_path, [oct_file(1:end-4) '_Profile_' suffix '.png']));
    slop=(Int(e)-Int(s))/(e-s);
% 
%     prompt = {'Enter IS band start point:','Enter IS band End point:'};
%     dlgtitle = 'Input';
%     dims = [1 35];
%     definput = {num2str(sini),num2str(eini)};
%     answer = inputdlg(prompt,dlgtitle,dims,definput);
%     s=str2double(answer{1});
%     sini=s;
%     e=str2double(answer{2});
%     eini=e;
% 
%     slop=(Int(e)-Int(s))/(e-s);

    while Int(s)+slop-Int(s+1)>0  && Int(s)-slop-Int(s-1)<0  && e < x_right + a_limit % adjust start point
         if e>s+1
            s=s+1;
            slop=(Int(e)-Int(s))/(e-s);
         else
             break
         end
    end

    while Int(s)+slop-Int(s+1)<0  && Int(s)-slop-Int(s-1)>0 && e < x_right + a_limit  % adjust start point
            s=s-1;
            slop=(Int(e)-Int(s))/(e-s);
     end

    while Int(e)+slop-Int(e+1)<0  && Int(e)-slop-Int(e-1)>0 && e < x_right + a_limit  % adjust start point
         if e>s+1
            e=e-1;
            slop=(Int(e)-Int(s))/(e-s);
         else
             break
         end
    end

    while Int(e)+slop-Int(e+1)>0  && Int(e)-slop-Int(e-1)<0  && e < x_right + a_limit % adjust start point
            e=e+1;
            slop=(Int(e)-Int(s))/(e-s);
     end

    IS=Int(s:e);
    
    f=figure(1);
    f.Position = [100 50 620 328];

    %Correct Baseline

    %Point at which graph will be rotated. We will be using the variable
    %"IS" as IS is the region of interest. We do not use the variable Int
    %as this variable contains the entire region of interest (ELM to RPE)

    x_rotation_point = (e + s)/2; %smidpoint of MCP/AR
    y_rotation_point = (Int(s) + Int(e)) / 2; %y-value at starting point

    %Rotation angle: we will be using cos(theta). So to calculate angle,
    %use inverse cosine
    adjacent_length = e - s;
    hypotenuse_length = sqrt( (e - s)^2 + (IS(end) - IS(1))^2 );

    theta_radians = acos(adjacent_length/hypotenuse_length);

    %Rotation matrix: clockwise
    R = [cos(-theta_radians) -sin(-theta_radians); 
         sin(-theta_radians) cos(-theta_radians)];
    
    %Shift points to origin
    number_of_points = max(size(Int(s:e)));
    x_points = linspace(s, e, number_of_points);
    y_points = Int(s:e)'; %originally (e - s) x 1 vector so need to make it 1 x (e - s)
    
    shifted_points = [x_points - x_rotation_point; y_points - y_rotation_point]; %Shift graph to origin

    %Rotate the graph
    rotated_points = R * shifted_points;

    %Shift points back to original center
    new_x = rotated_points(1, :) + x_rotation_point;
    new_y = rotated_points(2, :) + y_rotation_point;
    
    %Next step: given the new_y values, find the max value and
    %corresponding index and use this index on the original IS values to
    %get the max. Then rescale. (START HERE).
    
    [M,Index_max] = max(new_y);

    %Comparison to make sure rotated graph is truly rotated relative to
    %original IS. Can comment out this section once confirmed.
    % plot(new_x, new_y, 'b');
    % hold on
    % plot([new_x(1) new_x(end)], [new_y(1) new_y(end)], 'b');
    % plot(s:e, IS, 'r')
    % plot([s, e], [IS(1) IS(end)], 'r')
       
    %Rescaling. Once we get the maximum index, we use that index on the IS
    %region.
    Int_region = Int(s-3:e+5);
    max_Int_region = IS(Index_max);
    min_Int_region = min(IS);
    Int_region = (Int_region - min_Int_region ) / (max_Int_region  - min_Int_region);
    Int_region = Int_region * 100;
    
    plot((s-3:e+5),Int_region,'k')
    hold on
    plot([s,e],Int_region([4, end - 5]),'k')    
    set(gca,'xlim',[center-40 center+40])
    %set(gca,'Ylim',[-25 120])
    set(gca,'Ylim',[-20 120])
    
    saveas(gcf,fullfile(oct_path, [oct_file(1:end-4) '_ISez_' suffix '.png']));
    hold off

end