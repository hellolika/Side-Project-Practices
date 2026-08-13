CREATE PROCEDURE [dbo].[HRMS_RegisterMember.1.0.0]
	@username NVARCHAR(50),
	@email NVARCHAR(100),
	@password NVARCHAR(88),
	@phoneNumber NVARCHAR(30),
	@gender NVARCHAR(10),
	@address NVARCHAR(500),
	@salary DECIMAL,
	@permission INT,
	@bankAccount NVARCHAR(200),
	@isInProbation BIT,
	@remark NVARCHAR(500),
	@teamId INT,
	@jobGrade INT,
	@workLocationId INT,
	@isSupervisor BIT = 0,
	@joinDate DATETIME,
    @isAlertProbation BIT = 0,
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
	@IsCanSeeMemberSalary BIT
AS
BEGIN
	SET NOCOUNT ON;

	IF EXISTS (SELECT TOP 1 1 FROM [dbo].[Member] WITH(NOLOCK) WHERE [Username] = @username AND ([IsDeleted] = 0 OR [IsDeleted] IS NULL))
	BEGIN
		SELECT 25 AS ErrorCode;
		RETURN;
	END
	
	IF EXISTS (SELECT TOP 1 1 FROM [dbo].[Member] WITH(NOLOCK) WHERE [Email] = @email AND ([IsDeleted] = 0 OR [IsDeleted] IS NULL))
	BEGIN
		SELECT 26 AS ErrorCode;
		RETURN;
	END

	IF EXISTS (SELECT TOP 1 1 FROM [dbo].[Member] WITH(NOLOCK) WHERE [PhoneNumber] = @phoneNumber AND ([IsDeleted] = 0 OR [IsDeleted] IS NULL))
	BEGIN
		SELECT 34 AS ErrorCode;
		RETURN;
	END
	
	INSERT INTO [dbo].[Member] (
		[Username],
		[Email],
		[Password],
		[PhoneNumber],
		[Gender],
		[Address],
		[Salary],
		[Permission],
		[BankAccount],
		[IsInProbation],
		[Remark],
		[TeamId],
		[JobGrade],
		[WorkLocationId],
		[IsSupervisor],
		[JoinDate],
        [IsAlertProbation],
		[BankName],
		[PositionId],
        [Position],
        [EmployeeId],
		[IsFirstTimeUser],
		[Birthday],
		[NationalId],
		[VehicleType],
		[VehicleNumber],
		[DepartmentId],
		[IsManager],
		[IsCanSeeMemberSalary]
	) VALUES(
		@username,
		@email,
		@password,
		@phoneNumber,
		@gender,
		@address,
		@salary,
		@permission,
		@bankAccount,
		@isInProbation,
		@remark,
		@teamId,
		@jobGrade,
		@workLocationId,
		@isSupervisor,
		@joinDate,
        @isAlertProbation,
		@bankName,
		@positionId,
        @position,
        @employeeId,
		1,
		@birthday,
		@nationalId,
		@vehicleType,
		@vehicleNumber,
		@departmentId,
		@isManager,
		@IsCanSeeMemberSalary
	);
	SELECT 0 AS ErrorCode, [Id] AS MemberId from [dbo].[Member] WHERE [Username] = @username;
END
