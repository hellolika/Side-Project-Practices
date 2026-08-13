CREATE PROCEDURE [dbo].[HRMS_EditMember.1.0.1]
	@memberId INT,
	@username NVARCHAR(50),
	@email NVARCHAR(100),
	@phoneNumber NVARCHAR(30),
    @gender NVARCHAR(10),
	@address NVARCHAR(500),
	@salary DECIMAL,
	@permission INT,
	@bankAccount NVARCHAR(200),
	@isInProbation BIT,
	@isResigned BIT,
	@remark NVARCHAR(500),
	@teamId INT,
	@jobGrade INT,
	@workLocationId INT,
	@isSupervisor BIT = 0,
    @joinDate DATETIME,
    @isAlertProbation BIT,
    @bankName NVARCHAR(200),
	@positionId INT,
    @position NVARCHAR(200),
    @employeeId NVARCHAR(200),
	@birthday DATETIME,
	@nationalId NVARCHAR(200),
	@vehicleType NVARCHAR(200),
	@vehicleNumber NVARCHAR(200),
	@departmentId INT,
	@isManager BIT,
	@isCanSeeMemberSalary BIT
AS
BEGIN
	SET NOCOUNT ON;
	IF NOT EXISTS (SELECT TOP 1 1 FROM [dbo].[Member] WITH(NOLOCK) WHERE [Id] = @memberId)
	BEGIN
		SELECT 3 AS ErrorCode;
	END

	IF EXISTS (SELECT TOP 1 1 FROM [dbo].[Member] WITH(NOLOCK) WHERE [Username] = @username AND [Id] != @memberId)
	BEGIN
		SELECT 25 AS ErrorCode;
		RETURN;
	END
	
	IF EXISTS (SELECT TOP 1 1 FROM [dbo].[Member] WITH(NOLOCK) WHERE [Email] = @email AND [Id] != @memberId)
	BEGIN
		SELECT 26 AS ErrorCode;
		RETURN;
	END

	
	UPDATE [dbo].[Member] 
	SET [Username] = @username,
		[Email] = @email,
		[PhoneNumber] = @phoneNumber,
        [Gender] = @gender,
		[Address] = @address,
		[Salary] = CASE WHEN @salary = 0 THEN [Salary] ELSE @salary END,
		[Permission] = @permission,
		[BankAccount] = @bankAccount,
		[IsInProbation] = @isInProbation,
		[IsResigned] = @isResigned,
		[Remark] = @remark,
		[TeamId] = @teamId,
		[JobGrade] = @jobGrade,
		[WorkLocationId] = @workLocationId,
		[IsSupervisor] = @isSupervisor,
        [JoinDate] = @joinDate,
        [IsAlertProbation] = @isAlertProbation,
        [BankName] = @bankName,
		[PositionId] = @positionId,
        [Position] = @position,
        [EmployeeId] = @employeeId,
		[Birthday] = @birthday,
		[NationalId] = @nationalId,
		[VehicleType] = @vehicleType,
		[VehicleNumber] = @vehicleNumber,
		[DepartmentId] = @departmentId,
		[IsManager] = @isManager,
		[IsCanSeeMemberSalary] = @isCanSeeMemberSalary
	WHERE [Id] = @memberId

	SELECT 0 AS ErrorCode;
	
END