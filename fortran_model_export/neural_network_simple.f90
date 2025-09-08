program neural_network
    implicit none
    
    ! Network dimensions
    integer, parameter :: input_size = 2
    integer, parameter :: hidden_size = 8
    integer, parameter :: output_size = 1
    
    ! Network parameters
    real(8) :: ln1_weight(input_size), ln1_bias(input_size)
    real(8) :: linear_weight(hidden_size, input_size), linear_bias(hidden_size)
    real(8) :: ln2_weight(hidden_size), ln2_bias(hidden_size)
    real(8) :: output_weight(output_size, hidden_size), output_bias(output_size)
    
    ! Intermediate arrays
    real(8) :: input(input_size), normalized_input(input_size)
    real(8) :: hidden(hidden_size), normalized_hidden(hidden_size)
    real(8) :: output(output_size)
    
    ! Variables
    integer :: i, j, dims(3)
    real(8) :: mean_val, var_val
    real(8), parameter :: eps = 1e-5
    
    ! Read weights from simple text file
    open(unit=10, file='weights_simple.txt')
    
    read(10, *) dims
    
    ! Read all parameters in order
    do i = 1, input_size
        read(10, *) ln1_weight(i)
    end do
    do i = 1, input_size
        read(10, *) ln1_bias(i)
    end do
    do i = 1, hidden_size
        do j = 1, input_size
            read(10, *) linear_weight(i, j)
        end do
    end do
    do i = 1, hidden_size
        read(10, *) linear_bias(i)
    end do
    do i = 1, hidden_size
        read(10, *) ln2_weight(i)
    end do
    do i = 1, hidden_size
        read(10, *) ln2_bias(i)
    end do
    do j = 1, hidden_size
        read(10, *) output_weight(1, j)
    end do
    read(10, *) output_bias(1)
    
    close(10)
    
    ! Input values
    write(*,*) 'Enter two input values:'
    read(*,*) input(1), input(2)
    
    ! Forward pass
    ! 1. Input LayerNorm
    mean_val = sum(input) / input_size
    var_val = sum((input - mean_val)**2) / input_size
    normalized_input = (input - mean_val) / sqrt(var_val + eps)
    normalized_input = normalized_input * ln1_weight + ln1_bias
    
    ! 2. Linear layer
    do i = 1, hidden_size
        hidden(i) = 0.0d0
        do j = 1, input_size
            hidden(i) = hidden(i) + normalized_input(j) * linear_weight(i,j)
        end do
        hidden(i) = hidden(i) + linear_bias(i)
    end do
    
    ! 3. Hidden LayerNorm
    mean_val = sum(hidden) / hidden_size
    var_val = sum((hidden - mean_val)**2) / hidden_size
    normalized_hidden = (hidden - mean_val) / sqrt(var_val + eps)
    normalized_hidden = normalized_hidden * ln2_weight + ln2_bias
    
    ! 4. ReLU activation
    do i = 1, hidden_size
        normalized_hidden(i) = max(0.0d0, normalized_hidden(i))
    end do
    
    ! 5. Output layer
    do i = 1, output_size
        output(i) = 0.0d0
        do j = 1, hidden_size
            output(i) = output(i) + normalized_hidden(j) * output_weight(i,j)
        end do
        output(i) = output(i) + output_bias(i)
    end do
    
    ! 6. Tanh activation
    output(1) = tanh(output(1))
    
    ! Print result
    write(*,*) output(1)
    
end program neural_network