CREATE PROCEDURE [dbo].[HRMS_GetProfile.2.0.0]
	@memberId INT
AS
BEGIN
	SET NOCOUNT ON;

	IF NOT EXISTS (SELECT TOP 1 1 FROM [dbo].[Member] WHERE [Id] = @memberId AND ([IsDeleted] = 0 OR [IsDeleted] IS NULL))
	BEGIN
		SELECT 3 AS ErrorCode;
		RETURN;
	END

	SELECT 
		[Username], 
		[Email], 
		[PhoneNumber], 
		[Address], 
		[Salary], 
		[Permission], 
		[BankAccount], 
		[IsInProbation],[BankName], 
		[Position], 
		[EmployeeId], 
		[Remark], 
		m.[TeamId], 
		t.[TeamName],
		[JobGrade],
		j.[JobGradeName] as [JobGradeName],  
		[WorkLocationId], 
		l.[LocationName] AS [WorkLocation], 
		m.[Id] AS [MemberId], 
		[JoinDate],
		m.Birthday,
		m.NationalId,
		m.VehicleType,
		m.VehicleNumber,
		m.DepartmentId,
		d.[DepartmentName],
		m.IsManager,
		m.IsCanSeeMemberSalary
	FROM [dbo].[Member] m WITH(NOLOCK)
	INNER JOIN [dbo].[Team] t ON m.[TeamId] = t.[TeamId]
	INNER JOIN [dbo].[LocationDetail] l ON m.[WorkLocationId] = l.[Id]
	LEFT JOIN [dbo].[JobGrade] j ON j.[Id] = m.[JobGrade]
	LEFT JOIN [dbo].[Department] d ON d.[Id] = m.[DepartmentId]
	WHERE m.[Id] = @memberId
END