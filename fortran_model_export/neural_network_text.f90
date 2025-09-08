program neural_network
    implicit none
    
    ! Network dimensions
    integer, parameter :: input_size = 2
    integer, parameter :: hidden_size = 8
    integer, parameter :: output_size = 1
    
    ! Network parameters
    real(8) :: ln1_weight(input_size), ln1_bias(input_size)           ! Input LayerNorm
    real(8) :: linear_weight(hidden_size, input_size), linear_bias(hidden_size)  ! Linear layer
    real(8) :: ln2_weight(hidden_size), ln2_bias(hidden_size)         ! Hidden LayerNorm  
    real(8) :: output_weight(output_size, hidden_size), output_bias(output_size) ! Output layer
    
    ! Intermediate arrays
    real(8) :: input(input_size), normalized_input(input_size)
    real(8) :: hidden(hidden_size), normalized_hidden(hidden_size)
    real(8) :: output(output_size)
    
    ! Variables
    integer :: i, j, dims(3)
    real(8) :: mean_val, var_val
    real(8), parameter :: eps = 1e-5
    character(100) :: line
    
    ! Read weights from text file
    open(unit=10, file='actor_weights.txt')
    
    ! Skip comment and read dimensions
    read(10, '(A)') line
    read(10, *) dims
    read(10, '(A)') line  ! Skip empty line
    
    ! Read Input LayerNorm weights
    read(10, '(A)') line
    do i = 1, input_size
        read(10, *) ln1_weight(i)
    end do
    read(10, '(A)') line  ! Skip empty line
    
    ! Read Input LayerNorm biases
    read(10, '(A)') line
    do i = 1, input_size
        read(10, *) ln1_bias(i)
    end do
    
    ! Read Linear layer weights (row by row)
    read(10, '(A)') line
    do i = 1, hidden_size
        do j = 1, input_size
            read(10, *) linear_weight(i, j)
        end do
    end do
    
    ! Read Linear layer biases
    read(10, '(A)') line
    do i = 1, hidden_size
        read(10, *) linear_bias(i)
    end do
    
    ! Read Hidden LayerNorm weights
    read(10, '(A)') line
    do i = 1, hidden_size
        read(10, *) ln2_weight(i)
    end do
    
    ! Read Hidden LayerNorm biases
    read(10, '(A)') line
    do i = 1, hidden_size
        read(10, *) ln2_bias(i)
    end do
    
    ! Read Output layer weights
    read(10, '(A)') line
    do j = 1, hidden_size
        read(10, *) output_weight(1, j)
    end do
    
    ! Read Output layer bias
    read(10, '(A)') line
    read(10, *) output_bias(1)
    
    close(10)
    
    ! Input values
    write(*,*) 'Enter two input values:'
    read(*,*) input(1), input(2)
    write(*,*) 'Read input values:', input(1), input(2)
    
    ! Forward pass - matching PyTorch MLPActor structure
    
    ! 1. Input LayerNorm
    write(*,*) 'Step 1: Input LayerNorm'
    write(*,*) '  Input before norm:', input
    mean_val = sum(input) / input_size
    var_val = sum((input - mean_val)**2) / input_size
    write(*,*) '  Mean:', mean_val, '  Variance:', var_val
    normalized_input = (input - mean_val) / sqrt(var_val + eps)
    write(*,*) '  After standardization:', normalized_input
    normalized_input = normalized_input * ln1_weight + ln1_bias
    write(*,*) '  After LayerNorm (ln1):', normalized_input
    
    ! 2. Linear layer
    write(*,*) 'Step 2: Linear layer'
    write(*,*) '  First weight row:', linear_weight(1,:)
    write(*,*) '  First bias:', linear_bias(1)
    do i = 1, hidden_size
        hidden(i) = 0.0d0
        do j = 1, input_size
            hidden(i) = hidden(i) + normalized_input(j) * linear_weight(i,j)
        end do
        hidden(i) = hidden(i) + linear_bias(i)
    end do
    write(*,*) '  After linear layer:', hidden
    
    ! Manual verification for first output
    write(*,*) '  Manual check first output:'
    write(*,*) '    ', normalized_input(1), '*', linear_weight(1,1), '+', &
                     normalized_input(2), '*', linear_weight(1,2), '+', linear_bias(1)
    write(*,*) '    =', normalized_input(1)*linear_weight(1,1) + normalized_input(2)*linear_weight(1,2) + linear_bias(1)
    
    ! 3. Hidden LayerNorm
    write(*,*) 'Step 3: Hidden LayerNorm'
    mean_val = sum(hidden) / hidden_size
    var_val = sum((hidden - mean_val)**2) / hidden_size
    normalized_hidden = (hidden - mean_val) / sqrt(var_val + eps)
    normalized_hidden = normalized_hidden * ln2_weight + ln2_bias
    write(*,*) '  After LayerNorm (ln2):', normalized_hidden
    
    ! 4. ReLU activation
    write(*,*) 'Step 4: ReLU activation'
    do i = 1, hidden_size
        normalized_hidden(i) = max(0.0d0, normalized_hidden(i))
    end do
    write(*,*) '  After ReLU:', normalized_hidden
    
    ! 5. Output layer (linear)
    write(*,*) 'Step 5: Output layer'
    do i = 1, output_size
        output(i) = 0.0d0
        do j = 1, hidden_size
            output(i) = output(i) + normalized_hidden(j) * output_weight(i,j)
        end do
        output(i) = output(i) + output_bias(i)
    end do
    write(*,*) '  After output layer:', output
    
    ! 6. Tanh activation
    write(*,*) 'Step 6: Tanh activation'
    output(1) = tanh(output(1))
    write(*,*) '  After tanh:', output(1)
    
    ! Print result
    write(*,*) 'Final network output:', output(1)
    
end program neural_network