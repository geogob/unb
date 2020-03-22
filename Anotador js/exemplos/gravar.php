<?php
$sql = "INSERT INTO contas VALUES (?, ?, ?, ?)";

$id = $_POST['id'];
$pass = $_POST['pass'];
$data = date('Y-m-d');
$email = $_POST['email'];


$stmt = mysqli_prepare($con, $sql);
mysqli_stmt_bind_param($stmt, "isss", $id, $pass, $data, $email);
if(mysqli_stmt_execute($stmt)){
     echo 'registros inserido com sucesso';
}else{
     echo mysqli_error($con);
}

?>